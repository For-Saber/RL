
import numpy as np
import matplotlib.pyplot as plt


class BanditAgent:
    def __init__(self, num_arms=2):
        self.Q = [998.0] * num_arms  # 价值函数（你实验中已实现）
        self.epsilon = 0.1  # ε-greedy策略（调节探索与利用）

    def choose_action(self):  # 决策系统（智能体核心）
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.Q))  # 探索
        else:
            return np.argmax(self.Q)  # 利用

    def policy(self, observation):
        #W
        return

    def ucb_action(self, t):
        return np.argmax([q + np.sqrt(2 * np.log(t) / (n + 1e-5)) for q, n in zip(self.Q, self.N)])