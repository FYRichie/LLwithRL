import gym
from gym.envs.registration import register
import torch
from torch import random
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
            ac_losses,cr_losses = [], []
            total_rewards = []

            for episode in range(config["episode_per_batch"]):
                action_probs, benefit_degrees = [], []
                cr_ground_truth, cr = [], []
                state = self.env.reset()
                total_reward, total_step = 0, 0
                while True:
                    action, action_prob, cumulated_reward = self.actor_critic.sample(state)
                    next_state, reward, done, _ = self.env.step(action)
                    _, _, next_cumulated_reward = self.actor_critic.sample(next_state)

                    bd = reward + next_cumulated_reward - cumulated_reward
                    benefit_degrees.append(bd)  #list of tensor
                    action_probs.append(action_prob) #list of tensor
                    cr_ground_truth.append(reward)
                    cr.append(cumulated_reward) #list of tensor
                    state = next_state
                    total_reward += reward
                    total_step += 1
                    # print(len(benefit_degrees), len(action_probs), len(cr_ground_truth), len(cr))
                    if done or total_step == config["max_steps"]:
                        for i in range(len(cr_ground_truth) - 2, -1, -1):
                            cr_ground_truth[i] = cr_ground_truth[i + 1] * config["gamma"] + cr_ground_truth[i]
                        cr_loss = torch.stack([0.5 * (a - b)**2 for a, b in zip(cr, cr_ground_truth)])
                        ac_loss = torch.stack([-a * b for a, b in zip(action_probs, benefit_degrees)])
                        cr_losses.append(cr_loss) #list of tensor
                        ac_losses.append(ac_loss)
                        total_rewards.append(total_reward)
                        break
            self.actor_critic.learn(ac_losses, cr_losses)
            if config["save"] and batch % config["save_per_batch"] == 0:
                self.actor_critic.save(config["save_path"], batch)
            
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
        self.__training()
        # self.__get_trainig_result(avg_total_rewards, avg_final_rewards, "avg total rewards", "avg final rewards")
        # self.__get_trainig_result(plt_a_loss, plt_a_loss, "actor loss", "actor loss")
        # self.__get_trainig_result(plt_c_loss, plt_c_loss, "critic loss", "critic loss")
        self.__test_model()


if __name__ == "__main__":
    Main().main()