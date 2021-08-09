import gym
from gym.envs.registration import register
import torch
# from torch import random
from model import RLbase, Actor_Critic
from config import config
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Main():
    def __init__(self) -> None:
        self.env = gym.make("LunarLander-v2")
        self.base = RLbase()
        self.actor_critic = Actor_Critic(self.base)
        self.device = self.base.get_device()

    def __fix(self):
        self.env.seed(config["random_seed"])
        self.env.action_space.seed(config["random_seed"])
        torch.manual_seed(config["random_seed"])
        torch.cuda.manual_seed(config["random_seed"])
        torch.cuda.manual_seed_all(config["random_seed"])
        np.random.seed(config["random_seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def __set_environment(self):
        print(f'Random seed: {config["random_seed"]}')
        self.__fix()
        self.env.reset()
        print(f'Device: {self.device}')

    def __training(self):
        start_batch = 0
        if config["load"]:
            start_batch = self.actor_critic.load(config["load_path"])
        
        self.base.to(self.device)
        self.actor_critic.network.train()
        
        progress_bar = tqdm(range(start_batch, config["batch_num"]))
        for batch in progress_bar:
            action_probs, benefit_degrees = [], []
            cumulated_rewards = []

            for episode in range(config["episode_per_batch"]):
                cr_ground_truth, cr = [], []
                state = self.env.reset()
                total_reward, total_step = 0, 0
                while True:
                    action, action_prob, cumulated_reward = self.actor_critic.sample(state)
                    next_state, reward, done, _ = self.env.step(action)
                    _, _, next_cumulated_reward = self.actor_critic.sample(next_state)

                    bd = reward + next_cumulated_reward - cumulated_reward
                    benefit_degrees.append(bd)
                    action_probs.append(action_prob)
                    cr_ground_truth.append(reward)
                    cr.append(cumulated_reward)
                    state = next_state
                    total_reward += reward
                    total_step += 1
                    if done:
                        for i in range(len(cr_ground_truth) - 2, -1, -1):
                            cr_ground_truth[i] = cr_ground_truth[i + 1] * config["gamma"] + cr_ground_truth[i]
                        # cumulated_rewards.append(cr_ground_truth)
                        break
            
        
    def __get_trainig_result(self, avg_total_rewards, avg_final_rewards, label1, label2):
        plt.plot(avg_total_rewards, label=label1)
        plt.plot(avg_final_rewards, label=label2)
        plt.legend()
        plt.xlabel("batch num")
        plt.ylabel("reward")
        plt.title("Model training result")
        plt.show()

    def __test_model(self):
        self.__fix()
        self.actor_critic.network.eval()
        avg_reward = 0
        action_dist = {}
        for test_episode in range(config["test_episode_num"]):
            action_num, total_reward = 0, 0
            state = self.env.reset()
            done = False
            while not done:
                action, _, _ = self.actor_critic.sample(state)
                if action not in action_dist.keys():
                    action_dist[action] = 1
                else:
                    action_dist[action] += 1
                
                self.env.render()
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                action_num += 1
            print(f"Total reward: {total_reward: 4.1f}, Num of action: {action_num}")
            avg_reward += total_reward

        avg_reward = avg_reward / config["test_episode_num"]
        print(f"Model average reward when testing for {config['test_episode_num']} times is: %.2f"%avg_reward)
        print("Action distribution: ", action_dist)

    def main(self):
        self.__set_environment()
        avg_total_rewards, avg_final_rewards, plt_a_loss, plt_c_loss = self.__training()
        self.__get_trainig_result(avg_total_rewards, avg_final_rewards, "avg total rewards", "avg final rewards")
        self.__get_trainig_result(plt_a_loss, plt_a_loss, "actor loss", "actor loss")
        self.__get_trainig_result(plt_c_loss, plt_c_loss, "critic loss", "critic loss")
        self.__test_model()


if __name__ == "__main__":
    Main().main()