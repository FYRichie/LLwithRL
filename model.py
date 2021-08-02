import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from config import config

class RLbase(nn.Module):
    def __init__(self) -> None:
        super(RLbase, self).__init__()
        self.base_network = config["common_network"]
        self.actor_network = config["actor_network"]
        self.critic_network = config["critic_network"]
    
    def forward(self, x):
        x = self.base_network(x)
        return self.actor_network(x), self.critic_network(x)

    def get_actor_parameters(self):
        return [
            {"params": self.base_network.parameters()},
            {"params": self.actor_network.parameters(), **config["actor_optim_hparas"]}
        ]

    def get_critic_parameters(self):
        return [
            {"params": self.base_network.parameters()},
            {"params": self.critic_network.parameters(), **config["critic_optim_hparas"]}
        ]

class Actor():
    def __init__(self, base) -> None:
        self.network = base
        self.optimizer = optim.Adam(base.get_actor_parameters(), **config["common_optim_hparas"])

    def forward(self, x):
        x, _ = self.network(x)
        return x

    def sample(self, state):  # may add randomness to sampling
        action_prop = self.forward(torch.FloatTensor(state))
        action_dist = Categorical(action_prop)
        action = action_dist.sample()
        return action.item(), action_dist.entropy()

    def learn(self, cross_entropys, benefit_degrees):
        loss = (cross_entropys * benefit_degrees).sum()  # may use other definition, unsure about whether to add a "-" before calculating
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # TODO: Add save() and load()

class Critic():
    def __init__(self, base) -> None:
        self.network = base
        self.optimizer = optim.Adam(base.get_critic_parameters(), **config["common_optim_hparas"])

    def forward(self, x):
        _, x = self.network(x)
        return x

    def learn(self, cross_entropys, benefit_degrees):
        loss = (cross_entropys * benefit_degrees).sum()  # may use other definition, unsure about whether to add a "-" before calculating
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # TODO: Add save() and load()