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

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def forward(self, state):
        x = self.base_network(state)
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

    def save(self, batch, PATH):
        torch.save({
            "batch": batch
        }, PATH)
    
    def load(self, PATH):
        checkpoint = torch.load(PATH)
        return checkpoint["batch"]

class Actor():
    def __init__(self, base) -> None:
        self.network = base.to(base.get_device())
        # self.optimizer = optim.Adam(base.get_actor_parameters(), **config["common_optim_hparas"])
        self.optimizer = getattr(optim, config["actor_optimizer"])(base.get_actor_parameters(), **config["common_optim_hparas"])
        

    def forward(self, state):
        x, _ = self.network(state)
        return x

    def sample(self, state):  # may add randomness to sampling
        action_prop = self.forward(torch.FloatTensor(state))
        action_dist = Categorical(action_prop)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob()

    def learn(self, log_probs, benefit_degrees):
        loss = (log_probs * benefit_degrees).sum()  # may use other definition, unsure about whether to add a "-" before calculating
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, PATH):
        torch.save({
            "actor_network": self.network.state_dict(),
            "actor_optimizer": self.optimizer.state_dict()
        }, PATH)

    def load(self, PATH):
        checkpoint = torch.load(PATH)
        self.network.load_state_dict(checkpoint["actor_network"])
        self.optimizer.load_state_dict(checkpoint["actor_optimizer"])

class Critic():
    def __init__(self, base) -> None:
        self.network = base.to(base.get_device())
        # self.optimizer = optim.Adam(base.get_critic_parameters(), **config["common_optim_hparas"])
        self.optimizer = getattr(optim, config["critic_optimizer"])(base.get_critic_parameters(), **config["common_optim_hparas"])

    def forward(self, state):
        _, x = self.network(state)
        return x

    def learn(self, log_probs, benefit_degrees):
        loss = (log_probs * benefit_degrees).sum()  # may use other definition, unsure about whether to add a "-" before calculating
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, PATH):
        torch.save({
            "critic_network": self.network.state_dict(),
            "critic_optimizer": self.optimizer.state_dict()
        }, PATH)

    def load(self, PATH):
        checkpoint = torch.load(PATH)
        self.network.load_state_dict(checkpoint["critic_network"])
        self.optimizer.load_state_dict(checkpoint["critic_optimizer"])