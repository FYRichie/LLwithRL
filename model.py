import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from config import config

class RLbase(nn.Module):
    def __init__(self) -> None:
        super(RLbase, self).__init__()
        self.common_network = config["common_network"]
        self.actor_network = config["actor_network"]
        self.critic_network = config["critic_network"]

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, state):
        x = self.common_network(state)
        return self.actor_network(x), self.critic_network(x)

class Actor_Critic():
    def __init__(self, base) -> None:
        self.network = base
        self.optimizer = getattr(optim, config["optimizer"])(self.network.parameters(), **config["optim_hparas"])

    def sample(self, state):
        action_prop, cummulated_reward = self.network(torch.FloatTensor(state).to(self.network.get_device()))
        action_dist = Categorical(action_prop)
        action = action_dist.sample()
        return action.item(), action_dist[action], cummulated_reward

    def learn(self, actor_loss, critic_loss):
        loss = (actor_loss + critic_loss.pow(2)).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def save(self, PATH, progress):
        torch.save({
            "actor_critic_net": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "progress": progress
        }, PATH)

    def load(self, PATH):
        checkpoint = torch.load(PATH)
        self.network.load_state_dict(checkpoint["actor_critic_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint["progress"]