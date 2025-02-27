import numpy as np


class TicTacToeEnv:
    """井字棋环境（符合OpenAI Gym接口风格）"""

    def __init__(self):
        self.board = [' '] * 9  # 3x3棋盘[^2]
        self.current_player = 'X'

    def reset(self):
        self.board = [' '] * 9
        return self._get_state()

    def _get_state(self):
        return ''.join(self.board)  # 状态编码为字符串[^2]

    def step(self, action):
        """执行动作并返回（next_state, reward, done）"""
        if self.board[action] != ' ':
            return self._get_state(), -1, True  # 非法落子直接判负[^3]

        self.board[action] = self.current_player

        if self._check_win('X'):
            reward = 1 if self.current_player == 'X' else -1
            done = True
        elif ' ' not in self.board:  # 平局判断
            reward = 0
            done = True
        else:
            reward = 0
            done = False

        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return self._get_state(), reward, done

    def _check_win(self, player):
        # 检查所有获胜可能性
        win_states = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
            [0, 4, 8], [2, 4, 6]  # 对角线
        ]
        return any(all(self.board[i] == player for i in line) for line in win_states)


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9):
        self.q_table = {}  # 状态-动作价值表[^2]
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 未来奖励折扣

    def choose_action(self, state, valid_actions):
        """ε-greedy策略（ε=0.1）"""
        if np.random.rand() < 0.1 or state not in self.q_table:
            return np.random.choice(valid_actions)
        return valid_actions[np.argmax([self.q_table[state][a] for a in valid_actions])]

    def update_q(self, state, action, reward, next_state):
        """Q值更新公式"""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 9
        max_next = max(self.q_table[next_state]) if next_state in self.q_table else 0
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + self.alpha * (reward + self.gamma * max_next)

def train(episodes=20000):
    """训练过程"""
    env = TicTacToeEnv()
    agent = QLearningAgent()

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            valid_actions = [i for i, c in enumerate(env.board) if c == ' ']

            # 智能体行动（"X"玩家）
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done = env.step(action)

            # 对手行动（随机玩家）
            if not done:
                _, _, done = env.step(np.random.choice(valid_actions))

                # Q值更新
            agent.update_q(state, action, reward, next_state)
            state = next_state

        if (episode + 1) % 1000 == 0:
            print(f"完成训练局数: {episode + 1} / {episodes}, 已知状态数: {len(agent.q_table)}")

    return agent


def play_human_vs_agent(agent):
    """人类玩家对战AI"""
    env = TicTacToeEnv()
    state = env.reset()

    for _ in range(9):
        print("\n当前棋盘:")
        for i in range(0, 9, 3):
            print('|'.join(env.board[i:i + 3]))

        if env.current_player == 'X':  # 人类玩家回合
            while True:
                action = int(input("输入落子位置(0-8):"))
                if 0 <= action <= 8 and env.board[action] == ' ':
                    break
        else:  # AI回合
            valid_actions = [i for i, c in enumerate(env.board) if c == ' ']
            action = agent.choose_action(state, valid_actions)

        state, _, done = env.step(action)
        if done:
            print("游戏结束!")
            break


# 执行训练并测试
if __name__ == "__main__":
    print("开始训练...")
    trained_agent = train(episodes=5000)
    print("\n训练完成！最终掌握的棋局状态数量:", len(trained_agent.q_table))
    play_human_vs_agent(trained_agent)
