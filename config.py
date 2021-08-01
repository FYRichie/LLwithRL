import torch
import torch.nn as nn

config = {
    "actor_optimizer": "Adam",
    "critic_optimizer": "Adam",
    "common_network": nn.Sequential(
        # implement common network model
    ),
    "actor_network": nn.Sequential(
        # implement actor network model
        # actor is a classifier network
    ),
    "critic_network": nn.Sequential(
        # implement critic network model
        # critic is a regression network
    )
}