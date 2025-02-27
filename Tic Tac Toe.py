
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') # 或者尝试 'Qt5Agg'
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")



class TicTacToeEnv:
    """修正后的井字棋环境"""

    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 'X'

    def reset(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        return self._get_state()

    def _get_state(self):
        return ''.join(self.board)

    def step(self, action):
        if self.board[action] != ' ':
            return self._get_state(), -1, True  # 非法动作立即判负

        current_player = self.current_player
        self.board[action] = current_player

        # 修正胜负判断逻辑：检查当前玩家是否获胜
        if self._check_win(current_player):
            reward = 1 if current_player == 'X' else -1
            done = True
        elif ' ' not in self.board:  # 平局
            reward = 0
            done = True
        else:
            reward = 0
            done = False

        self.current_player = 'O' if current_player == 'X' else 'X'
        return self._get_state(), reward, done

    def _check_win(self, player):
        win_states = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
            [0, 4, 8], [2, 4, 6]  # 对角线
        ]
        return any(all(self.board[i] == player for i in line) for line in win_states)

    def get_valid_actions(self):
        return [i for i, c in enumerate(self.board) if c == ' ']


class QLearningAgent:
    """改进后的Q学习智能体"""

    def __init__(self, alpha=0.1, gamma=0.9):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, state, valid_actions):
        if not valid_actions:
            raise ValueError("No valid actions available")

        # ε-greedy策略
        if np.random.rand() < 0.1 or state not in self.q_table:
            return np.random.choice(valid_actions)

        q_values = [self.q_table[state][a] for a in valid_actions]
        return valid_actions[np.argmax(q_values)]

    def update_q(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 9

        max_next = max(self.q_table[next_state]) if next_state in self.q_table else 0
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
                                      self.alpha * (reward + self.gamma * max_next)


def train(episodes=30000):
    env = TicTacToeEnv()
    agent = QLearningAgent()
    vis = TrainingVisualizer()

    win_history = []
    step_history = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # 玩家X回合
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action = agent.choose_action(state, valid_actions)
            next_state, reward, done = env.step(action)

            # 立即更新Q表
            agent.update_q(state, action, reward, next_state)

            # 记录结果
            if done:
                win_history.append(1 if reward == 1 else 0)
                state = next_state
                break

            # 对手O回合
            valid_actions = env.get_valid_actions()
            opponent_action = np.random.choice(valid_actions)
            next_state, opponent_reward, done = env.step(opponent_action)

            # 使用对手回合的结果更新玩家Q表
            player_reward = -1 if opponent_reward == 1 else 0
            agent.update_q(state, action, player_reward, next_state)

            state = next_state
            steps += 1
            if done:
                win_history.append(1 if player_reward == 1 else 0)

        step_history.append(steps)

        # 每100局更新可视化
        if episode % 100 == 0:
            # 计算最近100局的胜率
            win_rate = np.mean(win_history[-100:]) if len(win_history) > 0 else 0
            # 计算最近100局的平均步数
            avg_steps = np.mean(step_history[-100:]) if len(step_history) > 0 else 0
            vis.update(episode, win_rate, len(agent.q_table), avg_steps)

    plt.ioff()
    plt.show()
    return agent


class TrainingVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.x_data = []
        self.win_rates = []
        self.q_sizes = []
        self.avg_steps = []

    def update(self, episode, win_rate, q_size, avg_step):
        self.x_data.append(episode)
        self.win_rates.append(win_rate)
        self.q_sizes.append(q_size)
        self.avg_steps.append(avg_step)

        self.ax1.clear()
        self.ax2.clear()

        # 胜率图表
        self.ax1.plot(self.x_data, self.win_rates, 'g-')
        self.ax1.set_ylabel('Win Rate')
        self.ax1.set_ylim(0, 1)
        self.ax1.grid(True)

        # Q表和步数图表
        self.ax2.plot(self.x_data, self.q_sizes, 'b-', label='Q-table Size')
        self.ax2.plot(self.x_data, self.avg_steps, 'r--', label='Average Steps')
        self.ax2.set_xlabel('Training Episodes')
        self.ax2.legend()
        self.ax2.grid(True)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

# 其余代码保持不变...

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
    trained_agent = train(episodes=300000)
    print("\n训练完成！最终掌握的棋局状态数量:", len(trained_agent.q_table))
    play_human_vs_agent(trained_agent)
