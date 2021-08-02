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
        fix(env=self.env, seed=config["random_seed"])
        self.env.reset()

    def __training(self):
        # implement Reinforcement Learning training algorithm
        self.actor.train()
        self.critic.train()

        avg_total_rewards, avg_final_rewards = [], []
        progress_bar = tqdm(range(config["batch_num"]))
        for batch in progress_bar:
            cross_entropys, benefit_degrees = [], []  # cross_entropys stores e_n, benefit_degrees stores A_n
            total_rewards, final_rewards = [], []  # total_rewards stores the total reward of the whole sequence, final_rewards stores the reward while finishing an episode(check to see if landing success)
            # collecting training data
            for episode in range(config["episode_per_batch"]):
                state = self.env.reset()
                total_reward, total_step = 0, 0
                while True:
                    action, entropy = self.actor.sample(state)
                    next_state, reward, done, _ = self.env.step(action)

                    bd = reward + self.critic.forward(next_state) - self.critic.forward(state)  # implements Advantage actor-critic
                    benefit_degrees.append(bd)
                    cross_entropys.append(entropy)
                    state = next_state
                    total_reward += reward
                    total_step += 1
                    if done:
                        final_rewards.append(reward)
                        total_rewards.append(total_reward)
                        break
            print(f"benefit degrees looks like ", np.shape(benefit_degrees))  
            print(f"cross entropys looks like ", np.shape(cross_entropys))
            # record training process
            avg_total_rewards.append(sum(total_rewards) / len(total_rewards))
            avg_final_rewards.append(sum(final_rewards) / len(final_rewards))
            progress_bar.set_description(f"Total: {avg_total_rewards[-1]: 4.1f}, Final: {avg_final_rewards[-1]: 4.1f}")
            # renew actor and critic
            benefit_degrees = (benefit_degrees - np.mean(benefit_degrees)) / (np.std(benefit_degrees) + 1e-9)  # standarize benefit degrees
            self.critic.learn(torch.stack(cross_entropys), torch.from_numpy(benefit_degrees))
            self.actor.learn(torch.stack(cross_entropys), torch.from_numpy(benefit_degrees))


    def main(args):
        # TODO: Finish main function
        pass


if __name__ == "__main__":
    Main().main()