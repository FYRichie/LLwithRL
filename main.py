import gym
from numpy.lib.function_base import average
import torch
from model import RLbase, Actor, Critic
from config import config
from fix import fix
import tqdm
import numpy as np

class Main():
    def __init__(self) -> None:
        self.env = gym.make("LunarLander-v2")
        self.base = RLbase()
        self.actor = Actor(self.base)
        self.critic = Critic(self.base)

    def __set_environment(self):
        print(f'Random seed: {config["random_seed"]}')
        fix(env=self.env, seed=config["random_seed"])
        self.env.reset()
        print(f"Device: {self.base.get_device()}")

    def __training(self):
        # implement Reinforcement Learning training algorithm
        self.actor.network.train()
        self.critic.network.train()

        device = self.base.get_device()
        avg_total_rewards, avg_final_rewards = [], []
        progress_bar = tqdm(range(config["batch_num"]))
        for batch in progress_bar:
            log_probs, benefit_degrees = [], []  # log_probs stores e_n, benefit_degrees stores A_n
            total_rewards, final_rewards = [], []  # total_rewards stores the total reward of the whole sequence, final_rewards stores the reward while finishing an episode(check to see if landing success)
            # collecting training data
            for episode in range(config["episode_per_batch"]):
                state = torch.tensor(self.env.reset()).to(device)
                total_reward, total_step = 0, 0
                while True:
                    action, log_prob = self.actor.sample(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = torch.tensor(next_state).to(device)

                    bd = reward + self.critic.forward(next_state) - self.critic.forward(state)  # implements Advantage actor-critic
                    benefit_degrees.append(bd)
                    log_probs.append(log_prob)
                    state = next_state
                    total_reward += reward
                    total_step += 1
                    if done:
                        final_rewards.append(reward)
                        total_rewards.append(total_reward)
                        break
            print(f"benefit degrees looks like ", np.shape(benefit_degrees))  
            print(f"cross log_probs looks like ", np.shape(log_probs))
            # record training process
            avg_total_rewards.append(sum(total_rewards) / len(total_rewards))
            avg_final_rewards.append(sum(final_rewards) / len(final_rewards))
            progress_bar.set_description(f"Total: {avg_total_rewards[-1]: 4.1f}, Final: {avg_final_rewards[-1]: 4.1f}")
            # renew actor and critic
            benefit_degrees = (benefit_degrees - np.mean(benefit_degrees)) / (np.std(benefit_degrees) + 1e-9)  # standarize benefit degrees
            log_probs = torch.stack(log_probs).to(device)
            benefit_degrees = torch.from_numpy(benefit_degrees).to(device)
            self.critic.learn(log_probs, benefit_degrees)
            self.actor.learn(log_probs, benefit_degrees)


    def main(self):
        # TODO: Finish main function
        self.__set_environment()
        pass


if __name__ == "__main__":
    Main().main()