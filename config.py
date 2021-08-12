import torch.nn as nn

config = {
    "common_network": nn.Sequential(
        # implement common network model
        # the last layer has the same dimension with actor network and critic network
        # the first layer dimension can be seen by calling env.observation_space
        nn.Linear(8, 64),
        nn.ReLU()
    ),
    "actor_network": nn.Sequential(
        # implement actor network model
        # actor is a classifier network
        # the last layer dimension can be seen by calling env.action_space
        nn.Linear(64, 4),
        nn.Softmax()
    ),
    "critic_network": nn.Sequential(
        # implement critic network model
        # critic is a regression network
        nn.Linear(64, 1),
    ),
    "optimizer": "Adam",
    "optim_hparas": {
        "lr": 0.002,
        "weight_decay": 1e-5,  # l2 regulization
        # "momentum": 0.5,  only needs when using SGD
    },
    "gamma": 0.99,  # discount factor
    "batch_num": 800,
    "max_steps": 600,
    "episode_per_batch": 5,
    "random_seed": 801,
    "load": True,
    "load_path": "./models/800batch600max.ckpt",
    "save": True,
    "save_per_batch": 10,
    "save_path": "./models/800batch600max.ckpt",
    "test_episode_num": 5,
}