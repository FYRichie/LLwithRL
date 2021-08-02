import torch
import torch.nn as nn

config = {
    "common_optim_hparas": {
        "lr": 0.0001,  # base learning rate
    },
    "actor_optimizer": "Adam",
    "actor_optim_hparas": {
        "lr": 0.0001,  # needs to be modified, can be different with base learning rate
        "weight_decay": 1e-5,  # optional while using Adam
        # "momentum": 0.5  # only need when actor optimizer using SGD
    },
    "critic_optimizer": "Adam",
    "critic_optim_hparas":{
        "lr": 0.0001,  # needs to be modified, can be different with base learning rate
        "weight_decay": 1e-5,  # optional while using Adam
        # "momentum": 0.5  #only need when critic optimizer using SGD
    },
    "common_network": nn.Sequential(
        # implement common network model
        # the last layer has the same dimension with actor network and critic network
    ),
    "actor_network": nn.Sequential(
        # implement actor network model
        # actor is a classifier network
    ),
    "critic_network": nn.Sequential(
        # implement critic network model
        # critic is a regression network
    ),
    # loss define: (-log_probs * rewards).sum(), may be modified
}