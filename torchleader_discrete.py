import imp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any
from collections import deque, namedtuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import OneHotEncoder
from multiclass import MulticlassToBandit

class Critic(nn.Module):
    def __init__(self, dim_context, dim_actions):
        super().__init__()
        self.embedding_dim = 32
        self.features_net = nn.Sequential(
            nn.Linear(dim_context+dim_actions, 256),
            nn.ReLU(),
            nn.Linear(256, self.embedding_dim),
            nn.ReLU(),
            )
        self.reward_net = torch.nn.Linear(self.embedding_dim, 1, bias=False)

    def features(self, x, a):
        emb = torch.cat((x,a),1)
        return self.features_net(emb)

    def forward(self, x, a):
        """
        X: contains concatenation of (state , 
        """
        phi = self.features(x, a)
        rew = self.reward_net(phi)
        return rew

Experience = namedtuple(
    "Experience",
    field_names=["context", "action", "reward"],
)

class SimpleBuffer:

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.
        """
        self.buffer.append(experience)

    def get_all(self):
        indices = np.arange(len(self.buffer))
        contexts, actions, rewards = zip(*(self.buffer[idx] for idx in indices))
        return (
            np.array(contexts),
            np.array(actions),
            np.array(rewards),
        )

@dataclass
class XBTorchDiscrete:

    env: Any
    net: nn.Module
    device: Optional[str]="cpu"
    batch_size: Optional[int]=256
    max_epochs: Optional[int]=1
    update_every_n_steps: Optional[int] = 100
    learning_rate: Optional[float]=0.001
    weight_l2param: Optional[float]=1. #weight_decay
    buffer_capacity: Optional[int]=10000
    seed: Optional[int]=0
    use_onehotencoding: Optional[bool]=False

    def reset(self):
        self.t = 1
        self.buffer = SimpleBuffer(capacity=self.buffer_capacity)
        self.instant_reward = np.zeros(1)
        self.best_reward = np.zeros(1)
        self.batch_counter = 0
        self.enc = None
        if self.use_onehotencoding:
            self.enc = OneHotEncoder(sparse=False)
            self.enc.fit(np.arange(self.env.action_space.n).reshape(-1,1))

    def _train_loss(self, b_context, b_actions, b_rewards):
        raise NotImplementedError

    def action(self, context: np.ndarray) -> int:
        raise NotImplementedError

    def _post_update(self, loader):
        pass

    def update(self, context: np.ndarray, action: int, reward: float):
        if self.enc:
            action = self.enc.transform(np.array([action]).reshape(-1,1)).ravel()
        exp = Experience(context, action, reward)
        self.buffer.append(exp)

        if self.t % self.update_every_n_steps == 0 and self.t > self.batch_size:
            contexts, actions, rewards = self.buffer.get_all()
            torch_dataset = torch.utils.data.TensorDataset(
                torch.tensor(contexts, dtype=torch.float, device=self.device), 
                torch.tensor(actions, dtype=torch.float, device=self.device),
                torch.tensor(rewards, dtype=torch.float, device=self.device)
                )

            loader = torch.utils.data.DataLoader(
                dataset=torch_dataset, 
                batch_size=self.batch_size, 
                shuffle=True
            )
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_l2param)
            for epoch in range(self.max_epochs):
                self.net.train()
                for b_context, b_actions, b_rewards in loader:
                    loss = self._train_loss(b_context, b_actions, b_rewards)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    self.writer.flush()
                    self.batch_counter += 1
            self.net.eval()

            self._post_update(loader)


    def _continue(self, horizon: int) -> None:
        """Continue learning from the point where we stopped
        """
        self.instant_reward = np.resize(self.instant_reward, horizon)
        self.best_reward = np.resize(self.best_reward, horizon)

    def run(self, horizon: int) -> None:
        self.writer = SummaryWriter(f"runs/{type(self).__name__}")

        self._continue(horizon)
        
        while (self.t < horizon):
            context = self.env.sample_context()
            action = self.action(context=context)
            reward = self.env.step(action)
            # update
            self.update(context, action, reward)
            # regret computation
            self.instant_reward[self.t] = self.env.expected_reward(action)
            self.best_reward[self.t] = self.env.best_reward()
            self.t += 1
        
        return {"regret": np.cumsum(self.best_reward - self.instant_reward)}
    

@dataclass
class TorchLeaderDiscrete(XBTorchDiscrete):

    noise_std: float=1
    features_bound: float=1
    param_bound: float=1
    delta: Optional[float]=0.01
    weight_mse: Optional[float]=1
    weight_spectral: Optional[float]=1
    weight_l2features: Optional[float]=1
    bonus_scale: Optional[float]=1.

    def __post_init__(self):
        self.inv_A = torch.eye(self.net.embedding_dim) / self.weight_l2param

    @torch.no_grad()
    def action(self, context: np.ndarray):
        dim = self.net.embedding_dim
        beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound*self.features_bound*self.t/self.weight_l2param)/self.delta)) + self.param_bound * np.sqrt(self.weight_l2param)

        # get features for each action and make it tensor
        
        na = self.env.action_space.n        
        if len(self.buffer) >= self.batch_size:
            contexts = torch.FloatTensor(np.tile(context, (na,1))).to(self.device)
            actions = np.arange(na).reshape(-1,1)
            if self.enc:
                actions = self.enc.transform(actions)
            actions = torch.FloatTensor(actions).to(self.device)

            prediction = self.net(contexts, actions).ravel()
            net_features = self.net.features(contexts, actions)
            #https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v/18542314#18542314
            ucb = ((net_features @ self.inv_A)*net_features).sum(axis=1)
            ucb = torch.sqrt(ucb)
            ucb = prediction + self.bonus_scale * beta * ucb
            action = torch.argmax(ucb).item()
            assert 0 <= action < na, ucb
        else:
            action = np.random.randint(0, na, 1).item()
        return action


    def _train_loss(self, b_context, b_actions, b_rewards):
        # MSE LOSS
        prediction = self.net(b_context, b_actions)
        mse_loss = F.mse_loss(prediction, b_rewards)
        self.writer.add_scalar('mse_loss', mse_loss, self.batch_counter)

        #DETERMINANT or LOG_MINEIG LOSS
        phi = self.net.features(b_context, b_actions)
        A = torch.sum(phi[...,None]*phi[:,None], axis=0)
        # det_loss = torch.logdet(A)
        spectral_loss = torch.log(torch.linalg.eigvalsh(A).min())
        self.writer.add_scalar('spectral_loss', spectral_loss, self.batch_counter)

        # FEATURES NORM LOSS
        l2feat_loss = torch.sum(torch.norm(phi, p=2, dim=1))
        # l2 reg on parameters can be done in the optimizer
        # though weight_decay (https://discuss.pytorch.org/t/simple-l2-regularization/139)
        self.writer.add_scalar('l2feat_loss', l2feat_loss, self.batch_counter)

        # TOTAL LOSS
        loss = self.weight_mse * mse_loss - self.weight_spectral * spectral_loss + self.weight_l2features * l2feat_loss
        self.writer.add_scalar('loss', loss, self.batch_counter)
        return loss
    
    def _post_update(self, loader):
        with torch.no_grad():
            dim = self.net.embedding_dim
            A = np.eye(dim) * self.weight_l2param
            for b_context, b_actions, b_rewards in loader:
                phi = self.net.features(b_context, b_actions).cpu().detach().numpy()
                A = A + np.sum(phi[...,None]*phi[:,None], axis=0)
            # strange issue with making operations directly in pytorch
            self.inv_A = torch.tensor(np.linalg.inv(A), dtype=torch.float)
            
@dataclass
class TorchLinUCBDiscrete(TorchLeaderDiscrete):

    def _train_loss(self, b_context, b_actions, b_rewards):
        # MSE LOSS
        prediction = self.net(b_context, b_actions)
        mse_loss = F.mse_loss(prediction, b_rewards)
        self.writer.add_scalar('mse_loss', mse_loss, self.batch_counter)
        return mse_loss
