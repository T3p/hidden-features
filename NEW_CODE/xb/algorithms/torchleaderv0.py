import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class TorchLeader:

    def __init__(
        self, env, representation, reg_val: float, noise_std: float,
        features_bound: float,
        param_bound: float, delta:float=0.01, random_state:int=0,
        device: str="cpu", batch_size:int=256, epochs:int=1,
        reg_mse: float=1, reg_spectral:float=1, reg_norm:float=1
    ) -> None:
        self.env = env
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
    
    def reset(self, horizon: int):
        self.t = 1
        dim = self.rep.features_dim()
        # we can replace this with buffer
        # and IterableDataset from pytorch
        self.X = np.zeros((horizon, dim))
        self.Y = np.zeros((horizon, 1))
        # self.Xc = np.zeros((horizon, 1))
        self.inv_A = torch.eye(dim, dtype=torch.float) / self.reg_val
        self.instant_reward = np.zeros(horizon)
        self.best_reward = np.zeros(horizon)
        self.uuu = 0

    @torch.no_grad()
    def action(self, context, available_actions):
        dim = self.X.shape[1]
        beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound*self.features_bound*self.t/self.reg_val)/self.delta)) + self.param_bound * np.sqrt(self.reg_val)

        n_arms = available_actions.shape[0]
        ref_feats = np.zeros((n_arms, dim))
        for i, a in enumerate(available_actions):
            v = self.rep.get_features(context, a)
            ref_feats[i] = v
        ref_feats = torch.tensor(ref_feats, dtype=torch.float, device=self.device)

        phi = self.net.features(
            torch.tensor(self.X[0:self.t], dtype=torch.float, device=self.device)).detach().numpy()
        A = phi.T @ phi + self.reg_val * np.eye(dim)
        inv_A = np.linalg.inv(A)

        val = self.net.predict(ref_feats).detach().numpy().ravel()
        gg = self.net.features(ref_feats).detach().numpy()
        ucb = np.einsum('...i,...i->...', gg @ inv_A, gg)
        ucb = np.sqrt(ucb)
        ucb = val + beta * ucb
        action = np.argmax(ucb)
        assert 0 <= action < n_arms, ucb
        return available_actions[action]

    def update(self, context, actions, rewards):
        v = self.rep.get_features(context, actions[0])
        self.X[self.t-1] = v
        self.Y[self.t-1] = rewards[0]
        # self.Xc[self.t-1] = context

        if self.t % 100 == 0:
            torch_dataset = torch.utils.data.TensorDataset(
                torch.tensor(self.X[0:self.t-1], dtype=torch.float, device=self.device), 
                torch.tensor(self.Y[0:self.t-1], dtype=torch.float, device=self.device)
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
                    A = phi.T @ phi
                    # det_loss = torch.logdet(A)
                    spectral_loss = torch.log(torch.linalg.eigvalsh(A).min())

                    # FEAT NORM LOSS
                    # fnorm_loss = torch.norm(phi, p=2, dim=1).max()
                    l2_loss = 0
                    for param in self.net.parameters():
                        l2_loss += torch.norm(param)

                    # TOTAL LOSS
                    loss = mse_loss * self.reg_mse + spectral_loss * self.reg_spectral + l2_loss * self.reg_norm

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
        dim = self.rep.features_dim()
        self.instant_reward.resize(horizon)
        self.best_reward.resize(horizon)
        self.X.resize((horizon, dim))
        self.Y.resize((horizon, 1))
        self.Xc.resize((horizon, 1))
    
    def run(self, horizon: int) -> None:

        self._continue(horizon)
        
        while (self.t < self.horizon):
            context = self.env.sample_context()
            avail_actions = self.env.get_available_actions()
            action = self.action(context=context, avail_actions=avail_actions)
            reward = self.env.step(action)

            # update
            self.update(context, action, reward)

            # regret computation
            self.instant_reward[self.t] = self.env.expected_reward(action)
            self.best_reward[self.t] = self.env.best_reward()
        
        return {"regret": np.cumsum(self.best_reward - self.instant_reward)}