import gym
from model import RLbase, Actor, Critic
from config import config
from fix import fix

class Main():
    def __init__(self) -> None:
        self.env = gym.make("LunarLander-v2")
        self.base = RLbase()
        self.actor = Actor(self.base)
        self.critic = Critic(self.base)

    def set_environment(self):
        fix(env=self.env, seed=config["random_seed"])
        self.env.reset()

    def training(self):
        pass

    def main():
        pass


if __name__ == "__main__":
    Main().main()