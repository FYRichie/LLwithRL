import torch.nn as nn

config = {
    "common_network": nn.Sequential(
        # implement common network model
        # the last layer has the same dimension with actor network and critic network
        # the first layer dimension can be seen by calling env.observation_space
        nn.Linear(8, 256),
        nn.ReLU()
    ),
    "actor_network": nn.Sequential(
        # implement actor network model
        # actor is a classifier network
        # the last layer dimension can be seen by calling env.action_space
        nn.Linear(256, 4),
        nn.Softmax()
    ),
    "critic_network": nn.Sequential(
        # implement critic network model
        # critic is a regression network
        nn.Linear(256, 1)
    ),
    "optimizer": "Adam",
    "optim_hparas": {
        "lr": 0.02,
        "weight_decay": 1e-5,  # l2 regulization
        # "momnetum": 0.5,  only needs when using SGD
    },
    "gamma": 0.99,  # discount factor
    "batch_num": 400,
    "max_steps": 999,
    "episode_per_batch": 5,
    "random_seed": 801,
    "load": False,
    "load_path": "",
    "save": False,
    "save_per_batch": 10,
    "save_path": "",
    "test_episode_num": 5,
}