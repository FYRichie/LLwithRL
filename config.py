import torch
import torch.nn as nn

config = {
    "common_optim_hparas": {
        "lr": 0.0001,  # base learning rate
        "weight_decay": 1e-5,  # optional while using Adam
    },
    "actor_optimizer": "Adam",
    "actor_optim_hparas": {
        "lr": 0.0001,  # needs to be modified, can be different with base learning rate
        "weight_decay": 1e-5,  # optional while using Adam
    },
    "critic_optimizer": "Adam",
    "critic_optim_hparas":{
        "lr": 0.0001,  # needs to be modified, can be different with base learning rate
        "weight_decay": 1e-5,  # optional while using Adam
    },
    "common_network": nn.Sequential(
        # implement common network model
        # the last layer has the same dimension with actor network and critic network
        # the first layer dimension can be seen by calling env.observation_space
    ),
    "actor_network": nn.Sequential(
        # implement actor network model
        # actor is a classifier network
        # the last layer dimension can be seen by calling env.action_space
    ),
    "critic_network": nn.Sequential(
        # implement critic network model
        # critic is a regression network
    ),
    "random_seed": 801,
    "batch_num": 400,  # times for actor, critic to renew
    "episode_per_batch": 5,  # the bigger the num is, the more training data can collect
}