import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from linearenv import LinearRepresentation

class XNet(nn.Module):
    def __init__(self, dim_input):
        """
        dim_input: dim_state + dim_action
        """
        super().__init__()
        self.dim_input = dim_input
        self._features = torch.nn.Linear(in_features=self.dim_input, out_features=self.dim_input)
        self._reward = torch.nn.Linear(self.dim_input, 1)

    def features(self, x):
        return self._features(x)

    def forward(self, x):
        """
        X: contains concatenation of (state , 
        """
        phi = self._features(x)
        rew = self._reward(phi)
        return rew

class SimpleCircData:

    def __init__(self, capacity:int, dim:int) -> None:
        self.capacity = capacity
        self.X = torch.zeros((capacity, dim), dtype=torch.float)
        self.Y = torch.zeros((capacity,1), dtype=torch.float)
        self.current_capacity = 0
        self.current_index = 0
    
    def __len__(self):
        return self.current_capacity
    
    def add(self, x, y):
        self.X[self.current_index] = x
        self.Y[self.current_index] = y
        self.current_index += 1
        self.current_index = self.current_index % self.capacity
        self.current_capacity = min(self.current_capacity+1, self.capacity)

    def get_all(self):
        if self.current_capacity == 0:
            return None
        elif self.current_capacity >= self.capacity:
            return self.X, self.Y
        else:
            return self.X[0:self.current_index], self.Y[0:self.current_index]


class TorchLeader:

    def __init__(
        self, env, 
        net: nn.Module,
        representation: LinearRepresentation, 
        noise_std: float,
        features_bound: float,
        param_bound: float, 
        delta:float=0.01, random_state:int=0,
        device: str="cpu", batch_size:int=256, max_epochs:int=1,
        weight_l2param: float=1.,
        weight_mse: float=1, weight_spectral:float=1, weight_l2features:float=1,
        buffer_capacity:int=10000,
        update_every_n_steps: int = 100,
        bonus_scale: float=1.,
        learning_rate: float=0.001
    ) -> None:
        self.env = env
        self.net = net
        self.rep = representation
        self.noise_std = noise_std
        self.features_bound = features_bound
        self.param_bound=param_bound
        self.delta = delta
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.device = device
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_l2param = weight_l2param
        self.weight_mse = weight_mse
        self.weight_spectral = weight_spectral
        self.weight_l2features = weight_l2features
        self.buffer_capacity = buffer_capacity
        self.update_every_n_steps = update_every_n_steps
        self.bonus_scale = bonus_scale
        self.learning_rate = learning_rate
    
    def reset(self):
        self.t = 1
        dim = self.rep.features_dim()
        self.buffer = SimpleCircData(capacity=self.buffer_capacity, dim=dim)
        self.inv_A = np.eye(dim, dtype=float) / self.weight_l2param
        self.instant_reward = np.zeros(1)
        self.best_reward = np.zeros(1)
        self.batch_counter = 0

    @torch.no_grad()
    def action(self, context: np.ndarray, available_actions: np.ndarray):
        dim = self.rep.features_dim()
        beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound*self.features_bound*self.t/self.weight_l2param)/self.delta)) + self.param_bound * np.sqrt(self.weight_l2param)

        # get features for each action and make it tensor
        n_arms = available_actions.shape[0]
        ref_feats = np.zeros((n_arms, dim))
        for i, a in enumerate(available_actions):
            v = self.rep.get_features(context, a)
            ref_feats[i] = v
        ref_feats = torch.tensor(ref_feats, dtype=torch.float, device=self.device)

        if len(self.buffer) >= self.batch_size:
            prediction = self.net(ref_feats).detach().numpy().ravel()
            net_features = self.net.features(ref_feats).detach().numpy()
            ucb = np.einsum('...i,...i->...', net_features @ self.inv_A, net_features)
            ucb = np.sqrt(ucb)
            ucb = prediction + self.bonus_scale * beta * ucb
            action = np.argmax(ucb)
            assert 0 <= action < n_arms, ucb
        else:
            action = np.random.choice(n_arms, 1).item()
        return action

    def update(self, context: np.ndarray, action: int, reward: float):
        v = self.rep.get_features(context, action)
        self.buffer.add(torch.FloatTensor(v), torch.FloatTensor(reward))

        if self.t % 100 == 0 and self.t > self.batch_size:
            X, Y = self.buffer.get_all()
            torch_dataset = torch.utils.data.TensorDataset(
                X.to(self.device), 
                Y.to(self.device)
                )

            loader = torch.utils.data.DataLoader(
                dataset=torch_dataset, 
                batch_size=self.batch_size, 
                shuffle=True
            )
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_l2param)
            for epoch in range(self.max_epochs):
                self.net.train()
                for x, y in loader:# MSE LOSS
                    prediction = self.net(x)
                    mse_loss = F.mse_loss(prediction, y)
                    self.writer.add_scalar('mse_loss', mse_loss, self.batch_counter)

                    #DETERMINANT or LOG_MINEIG LOSS
                    phi = self.net.features(x)
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

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    self.writer.flush()
                    self.batch_counter += 1
            self.net.eval()

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
            avail_actions = self.env.get_available_actions()
            action = self.action(context=context, available_actions=avail_actions)
            reward = self.env.step(action)

            # update
            self.update(context, action, reward)

            # regret computation
            self.instant_reward[self.t] = self.env.expected_reward(action)
            self.best_reward[self.t] = self.env.best_reward()

            self.t += 1
        
        return {"regret": np.cumsum(self.best_reward - self.instant_reward)}