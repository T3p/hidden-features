import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

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

    def predict(self, x):
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
        self, env, net, representation, reg_val: float, noise_std: float,
        features_bound: float,
        param_bound: float, delta:float=0.01, random_state:int=0,
        device: str="cpu", batch_size:int=256, epochs:int=1,
        reg_mse: float=1, reg_spectral:float=1, reg_norm:float=1,
        bonus_scale:float=1, buffer_capacity:int=100
    ) -> None:
        assert reg_mse >= 0 and reg_spectral >= 0 and reg_norm >= 0
        self.env = env
        self.net = net
        self.rep = representation
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.features_bound = features_bound
        self.param_bound=param_bound
        self.delta = delta
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_mse = reg_mse
        self.reg_spectral = reg_spectral
        self.reg_norm = reg_norm
        self.bonus_scale=bonus_scale
        self.buffer_capacity = buffer_capacity
    
    def reset(self):
        self.t = 1
        dim = self.rep.features_dim()
        # we can replace this with buffer
        # and IterableDataset from pytorch
        self.buffer = SimpleCircData(capacity=self.buffer_capacity, dim=dim)
        self.inv_A = torch.eye(dim, dtype=torch.float) / self.reg_val
        self.instant_reward = np.zeros(1)
        self.best_reward = np.zeros(1)
        self.uuu = 0

    @torch.no_grad()
    def action(self, context: np.ndarray, available_actions: np.ndarray):
        dim = self.rep.features_dim()
        beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound*self.features_bound*self.t/self.reg_val)/self.delta)) + self.param_bound * np.sqrt(self.reg_val)

        n_arms = available_actions.shape[0]
        ref_feats = np.zeros((n_arms, dim))
        for i, a in enumerate(available_actions):
            v = self.rep.get_features(context, a)
            ref_feats[i] = v
        ref_feats = torch.tensor(ref_feats, dtype=torch.float, device=self.device)

        if len(self.buffer) >= self.batch_size:
            X, _ = self.buffer.get_all()
            phi = self.net.features(X.to(self.device)).detach().numpy()
            A = phi[...,None]*phi[:,None]
            A = np.sum(A, axis=0) + self.reg_val * np.eye(dim)
            inv_A = np.linalg.inv(A)

            val = self.net.predict(ref_feats).detach().numpy().ravel()
            gg = self.net.features(ref_feats).detach().numpy()
            ucb = np.einsum('...i,...i->...', gg @ inv_A, gg)
            ucb = np.sqrt(ucb)
            ucb = val + self.bonus_scale * beta * ucb
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
            optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
            loss_func = torch.nn.MSELoss()
            for epoch in range(self.epochs):
                steps = 0
                tot = 0
                tot_mse = 0
                tot_spec = 0
                tot_l2loss = 0
                for batch_x, batch_y in loader:
                    # MSE LOSS
                    prediction = self.net.predict(batch_x)
                    mse_loss = loss_func(prediction, batch_y)

                    #DETERMINANT or LOG_MINEIG LOSS
                    phi = self.net.features(batch_x)
                    A = torch.sum(phi[...,None]*phi[:,None], axis=0)
                    # det_loss = torch.logdet(A)
                    spectral_loss = torch.log(torch.linalg.eigvalsh(A).min())

                    # FEAT NORM LOSS
                    # fnorm_loss = torch.norm(phi, p=2, dim=1).max()
                    l2_loss = 0
                    for param in self.net.parameters():
                        l2_loss += torch.norm(param)

                    # TOTAL LOSS
                    loss = mse_loss * self.reg_mse - spectral_loss * self.reg_spectral + l2_loss * self.reg_norm

                    optimizer.zero_grad()   # clear gradients for next train
                    loss.backward()         # backpropagation, compute gradients
                    optimizer.step()        # apply gradients
                    tot += loss.item()
                    tot_mse += mse_loss.item()
                    tot_spec += spectral_loss.item()
                    tot_l2loss += l2_loss.item() 
                    steps += 1
                self.writer.add_scalar('loss', tot/steps, self.uuu)
                self.writer.add_scalar('mse', tot_mse/steps, self.uuu)
                self.writer.add_scalar('spectral_loss', tot_spec/steps, self.uuu)
                self.writer.add_scalar('l2loss', tot_l2loss/steps, self.uuu)
                self.uuu += 1
                self.writer.flush()


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