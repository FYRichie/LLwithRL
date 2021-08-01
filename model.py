import torch
import torch.nn as nn
from config import config

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.__common_network = config.common_network
        self.__actor_network = config.actor_network
        self.__critic_network = config.critic_network
        
    def forward(self, x):
        # unsure about what to do with critic network
        pass