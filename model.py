import torch
import torch.nn as nn
import torch.optim as optim
from config import config

class RLbase(nn.Module):
    def __init__(self) -> None:
        super(RLbase, self).__init__()
        self.base_network = config.common_network
        self.actor_network = config.actor_network
        self.critic_network = config.critic_network
    
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

class Critic():
    def __init__(self, base) -> None:
        self.network = base
        self.optimizer = optim.Adam(base.get_critic_parameters(), **config["common_optim_hparas"])

    def forward(self, x):
        _, x = self.network(x)
        return x