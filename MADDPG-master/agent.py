import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    # def select_action(self, o, noise_rate, epsilon):
    #     if np.random.uniform() < epsilon:
    #         u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
    #     else:
    #         inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
    #         pi = self.policy.actor_network(inputs).squeeze(0)
    #         # print('{} : {}'.format(self.name, pi))
    #         u = pi.cpu().numpy()
    #         noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
    #         u += noise
    #         u = np.clip(u, -self.args.high_action, self.args.high_action)
    #     return u.copy()

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            # 随机选择 1~5 的整数
            u = np.random.randint(0, 5, self.args.action_shape[self.agent_id])  # 1~5（包含）
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(inputs).squeeze(0)
            u = pi.cpu().detach().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            # 裁剪到 [1, 5] 范围，并四舍五入为整数
            u = np.clip(u, 0, 4)
            u = np.round(u).astype(int)
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

