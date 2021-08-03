import gym
import torch
from model import RLbase, Actor, Critic
from config import config
from fix import fix
from tqdm._tqdm_notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Main():
    def __init__(self) -> None:
        self.env = gym.make("LunarLander-v2")
        self.base = RLbase()
        self.actor = Actor(self.base)
        self.critic = Critic(self.base)
        self.device = self.base.get_device()

    def __set_environment(self):
        print(f'Random seed: {config["random_seed"]}')
        fix(env=self.env, seed=config["random_seed"])
        self.env.reset()
        print(f"Device: {self.device}")

    def __training(self):
        start_batch = 0
        if config["load"]:
            start_batch = self.base.load(config["load_path"])
            self.actor.load(config["load_path"])
            self.critic.load(config["load_path"])

        # implement Reinforcement Learning training algorithm
        self.actor.network.train()
        self.critic.network.train()

        avg_total_rewards, avg_final_rewards = [], []
        progress_bar = tqdm(range(start_batch, config["batch_num"]))
        for batch in progress_bar:
            log_probs, benefit_degrees = [], []  # log_probs stores e_n(here for log prob of prediction), benefit_degrees stores A_n
            total_rewards, final_rewards = [], []  # total_rewards stores the total reward of the whole sequence, final_rewards stores the reward while finishing an episode(check to see if landing success)
            # collecting training data
            for episode in range(config["episode_per_batch"]):
                state = torch.tensor(self.env.reset()).to(self.device)
                total_reward, total_step = 0, 0
                while True:
                    action, log_prob = self.actor.sample(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = torch.tensor(next_state).to(self.device)

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
            print(f"cross entropys looks like ", np.shape(log_probs))
            # record training process
            avg_total_rewards.append(sum(total_rewards) / len(total_rewards))
            avg_final_rewards.append(sum(final_rewards) / len(final_rewards))
            progress_bar.set_description(f"Total: {avg_total_rewards[-1]: 4.1f}, Final: {avg_final_rewards[-1]: 4.1f}")
            # renew actor and critic
            benefit_degrees = (benefit_degrees - np.mean(benefit_degrees)) / (np.std(benefit_degrees) + 1e-9)  # standarize benefit degrees
            log_probs = torch.stack(log_probs).to(self.device)
            benefit_degrees = torch.from_numpy(benefit_degrees).to(self.device)
            self.critic.learn(benefit_degrees)
            self.actor.learn(log_probs, benefit_degrees)
            # save model if needed
            if config["save"] and batch % config["save_per_batch"] == 0:
                self.base.save(batch, config["save_path"])
                self.actor.save(config["save_path"])
                self.critic.save(config["save_path"])
        return avg_total_rewards, avg_final_rewards

    def __get_trainig_result(self, avg_total_rewards, avg_final_rewards):
        plt.plot(avg_total_rewards, label="avg total rewards")
        plt.plot(avg_final_rewards, label="avg final rewards")
        plt.xlabel("batch num")
        plt.ylabel("reward")
        plt.title("Model training result")
        plt.show()

    def __test_model(self):
        fix(self.env, config["random_seed"])
        self.actor.network.eval()
        # self.critic.network.eval()
        avg_reward = 0
        action_distribution = {}
        for test_episode in range(config["test_episode_num"]):
            action_num, total_reward = 0, 0
            state = torch.tensor(self.env.reset()).to(self.device)
            done = False
            while not done:
                action, _ = self.actor.sample(state)
                if action not in action_distribution.keys():
                    action_distribution[action] = 1
                else:
                    action_distribution[action] += 1
                
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                action_num += 1
            print(f"Total reward: {total_reward: 4.1f}, Num of action: {action_num}")
            avg_reward += total_reward

        avg_reward = avg_reward / config["test_episode_num"]
        print(f"Model average reward when testing for {config['test_episode_num']} times is: %.2f"%avg_reward)
        print("Action distribution: ", action_distribution)

    def main(self):
        # TODO: Finish main function
        self.__set_environment()
        avg_total_rewards, avg_final_rewards = self.__training()
        self.__get_trainig_result(avg_total_rewards, avg_final_rewards)


if __name__ == "__main__":
    Main().main()