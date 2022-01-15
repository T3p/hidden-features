import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import random


class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def push(self, sample):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(len(self.memory), batch_size)

        ############################## REPLACE THIS WITH NUMPY!!!!!!!!!!!!!
        samples = random.sample(self.memory, batch_size)
        return map(np.asarray, zip(*samples))

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = []
        self.position = 0

class RegBasedCBDiscrete:
    def __init__(
        self, model_constructor, optimizer_constructor, 
        memory_capacity, epochs=20, batch_size=64, device="cpu", seed=0,
        exploration="egreedy"
    ) -> None:
        self.model_constructor = model_constructor
        self.optimizer_constructor = optimizer_constructor
        # self.memory_capacity = memory_capacity
        self.buffer = ReplayBuffer(capacity=memory_capacity, seed=seed)
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.device = device
        self.epochs = epochs
        assert exploration in ["egreedy", "igw", "softmax"]
        self.exploration = exploration
        # self.n_actions = n_actions
    
    def reset(self):
        self.t = 1
        self.buffer.reset()
        self.model = self.model_constructor()
        if self.optimizer_constructor is None:
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=0.1)
        else:
            self.optimizer = self.optimizer_constructor(self.model)
    
    def action(self, context, available_actions):
        
        X = torch.FloatTensor(context['values'].reshape(1,-1)).to(self.device)
        values = self.model(X)
        
        if self.rng.rand() < 0.1:
            action = self.rng.choice(len(available_actions), 1).item()
        else:
            action = torch.argmax(values, dim=1).item()
        return available_actions[action]
            
    def update(self, context, action, reward):
        self.t += 1
        self.buffer.push((context['values'], action['id'], reward))

        if len(self.buffer) >= self.batch_size:

            for k in range(self.epochs):
                batch_context, batch_action, batch_reward = self.replay_buffer.sample(
                    self.batch_size
                )

                batch_context = torch.FloatTensor(batch_context).to(self.device)
                batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(self.device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)

                values = self.model(batch_context).gather(1, batch_action.long())

                loss = F.mse_loss(values, batch_reward)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


class RegBasedCBFeatures(RegBasedCBDiscrete):

    def __init__(
        self, rep, model_constructor, optimizer_constructor, 
        memory_capacity, epochs=20, batch_size=64, device="cpu", seed=0,
        exploration="egreedy",
        igw_forcing=None, igw_mult=None
    ) -> None:
        super().__init__(
            model_constructor, optimizer_constructor, memory_capacity=memory_capacity, 
            epochs=epochs, batch_size=batch_size, device=device, seed=seed, exploration=exploration
        )
        self.rep = rep
        self.igw_forcing = igw_forcing
        self.igw_mult = igw_mult
    
    def action(self, context, available_actions):
        na = len(available_actions)
        dim = self.rep.features_dim()
        X = np.zeros((na, dim))
        for i in range(na):
            v = self.rep.get_features(context, available_actions[i])
            X[i] = v
        X = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            values = self.model(X)
        
        if self.exploration == "egreedy":
            if self.rng.rand() < 1./self.t:
                action = self.rng.choice(na, 1).item()
            else:
                action = torch.argmax(values).item()
        elif self.exploration == "softmax":
            action = torch.softmax(values * self.t, dim=0)
            action = torch.multinomial(action.ravel(), 1).item()
        elif self.exploration == "igw":
            action = torch.argmax(values, dim=0).item()
            p = np.zeros(na)
            for i in range(na):
                if i != action:
                    p[i] = 1. / (self.igw_forcing + self.igw_mult * np.sqrt(self.t) * (values[action] - values[i]))
            p[action] = 1. - np.sum(p)
            action = self.rng.choice(na, 1, p=p).item()
        return available_actions[action]
            
    def update(self, context, action, reward):
        self.t += 1
        v = self.rep.get_features(context, action)
        self.buffer.push((v, reward))

        if len(self.buffer) >= self.batch_size:

            for k in range(self.epochs):
                batch_features, batch_reward = self.buffer.sample(self.batch_size)

                batch_features = torch.FloatTensor(batch_features).to(self.device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)

                values = self.model(batch_features)

                loss = F.mse_loss(values, batch_reward)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.t % 1000 == 0:
                print(f"{self.t} - loss: {loss.item()}")

# class RegBasedCBHotEnc:

#     def __init__(self, model_constructor, optimizer_constructor, memory_capacity, n_actions, batch_size=64, device="cpu", seed=0) -> None:
#         self.model_constructor = model_constructor
#         self.optimizer_constructor = optimizer_constructor
#         self.buffer = ReplayBuffer(capacity=memory_capacity, seed=seed)
#         self.batch_size = batch_size
#         self.seed = seed
#         self.rng = np.random.RandomState(seed)
#         self.device = device
#         self.n_actions = n_actions
    
#     def reset(self):
#         self.t = 1
#         self.buffer.reset()
#         self.enc = OneHotEncoder(
#                 categories=list(range(self.n_actions)),
#                 handle_unknown='error'
#             )
#         self.enc.fit(np.arange(self.n_actions).reshape(-1,1))
#         if self.optimizer_constructor is None:
#             self.optimizer = optim.Adam(params=self.model.parameters(), lr=0.1)
#         else:
#             self.optimizer = self.optimizer_constructor(self.model)
#         self.model = self.model_constructor()

#     def action(self, context, available_actions):
        
#         na = len(available_actions)
#         context = context.reshape(1,-1)
#         actions = np.array((na, 1))
#         states = np.array((na, context.shape[1]))
#         for i in range(na):
#             actions[i] = available_actions[i]['id']
#             states[i] = context
#         actions = self.enc.transform(actions)
#         X = np.concatenate([states, actions], axis=1)
#         values = self.model(X).cpu().detach().numpy()
        
#         if self.rng.rand() < 0.1:
#             action = self.rng.choice(na, 1).item()
#         else:
#             action = np.argmax(values)
#         return available_actions[action]
            
#     def update(self, context, action, reward):
#         action_tostore = self.enc.transform(np.array([action['id']]).reshape(1,-1))
#         self.buffer.push((context, action_tostore, reward))

#         if len(self.buffer) >= self.batch_size:

#             batch_context, batch_action, batch_reward = self.replay_buffer.sample(self.batch_size)

#             batch_context = torch.FloatTensor(batch_context).to(self.device)
#             batch_action = torch.FloatTensor(batch_action).to(self.device)
#             batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)

#             values = 0

#             loss = F.mse_loss(values, batch_reward)
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()


