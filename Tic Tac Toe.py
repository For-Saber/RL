import numpy as np
import matplotlib.pyplot as plt

class TicTacToeEnv:
    """井字棋环境（符合OpenAI Gym接口风格）"""

    def __init__(self):
        self.board = [' '] * 9  # 3x3棋盘
        self.current_player = 'X'

    def reset(self):
        self.board = [' '] * 9
        return self._get_state()

    def _get_state(self):
        return ''.join(self.board)  # 状态编码为字符串

    def step(self, action):
        """执行动作并返回（next_state, reward, done）"""
        if self.board[action] != ' ':
            return self._get_state(), -1, True  # 非法落子直接判负

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

    def get_valid_actions(self):
        """安全获取有效动作列表，附带有效性检查"""
        actions = [i for i, c in enumerate(self.board) if c == ' ']
        if len(actions) == 0:
            self.done = True  # 自动设置终止标志
        return actions

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9):
        self.q_table = {}  # 状态-动作价值表[^2]
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 未来奖励折扣

    def choose_action(self, state, valid_actions):

        if not valid_actions:
            raise EnvironmentError("Cannot choose action when no valid actions exist")
        #ε-greedy策略（ε=0.1）
        if np.random.rand() < 0.1 or state not in self.q_table:
            return np.random.choice(valid_actions)
        return valid_actions[np.argmax([self.q_table[state][a] for a in valid_actions])]

    def update_q(self, state, action, reward, next_state):
        #Q值更新公式
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 9
        max_next = max(self.q_table[next_state]) if next_state in self.q_table else 0
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + self.alpha * (reward + self.gamma * max_next)


class TrainingVisualizer:
    """训练过程可视化模块"""

    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.win_rates = []
        self.q_sizes = []
        self.steps = []

    def update(self, episode, wins, q_size, steps):
        """更新绘图数据"""
        # 计算胜率（滑动窗口取最近100局的胜率）[^2]
        window_size = 100
        recent_wins = wins[-window_size:]
        win_rate = sum(recent_wins) / len(recent_wins) if len(recent_wins) > 0 else 0

        self.win_rates.append(win_rate)
        self.q_sizes.append(q_size)
        self.steps.append(steps)

        # 清空画布
        self.ax1.clear()
        self.ax2.clear()

        # 绘制胜率曲线
        self.ax1.plot(self.win_rates, 'g-', label='Win Rate (100-episode avg)')
        self.ax1.set_ylabel('Win Rate')
        self.ax1.set_ylim(0, 1)
        self.ax1.legend()

        # 绘制Q表大小和平均步数曲线
        self.ax2.plot(self.q_sizes, 'b-', label='Q-table Size')
        self.ax2.plot(self.steps, 'r--', label='Average Steps')
        self.ax2.set_xlabel('Training Episodes')
        self.ax2.legend()

        plt.pause(0.01)  # 轻微刷新图表


def train(episodes=20000):
    """改进后的训练函数（含可视化）"""
    env = TicTacToeEnv()
    agent = QLearningAgent()
    vis = TrainingVisualizer()

    total_wins = []  # 记录所有胜负结果（1=胜利，0=失败或平局）
    avg_steps = []
    steps_buffer = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done:
            # 玩家行动 ---------------------------------------------------
            valid_actions_player = env.get_valid_actions()  # 新增辅助方法[^3]
            if not valid_actions_player:  # 保护性判断
                break
            action = agent.choose_action(state, valid_actions_player)
            next_state, reward, done = env.step(action)
            agent.update_q(state, action, reward, next_state)
            state = next_state

            if done:  # 玩家行动终结棋局时直接退出
                break

            # 对手行动 ---------------------------------------------------
            valid_actions_opponent = env.get_valid_actions()
            opponent_action = np.random.choice(valid_actions_opponent)
            next_state, _, done = env.step(opponent_action)  # 必须获取对手行动后的状态
            state = next_state  # 更新状态

            agent.update_q(state, action, reward, next_state)
            state = next_state
            steps += 1

        # 记录结果
        total_wins.append(1 if reward == 1 else 0)
        steps_buffer.append(steps)
        if (episode + 1) % 100 == 0:  # 每100局计算平均步数[^2]
            avg_steps.append(np.mean(steps_buffer))
            steps_buffer = []

        # 更新可视化（每100局）
        if (episode + 1) % 100 == 0:
            vis.update(episode + 1, total_wins, len(agent.q_table), avg_steps[-1] if len(avg_steps) > 0 else 0)

    plt.show()  # 训练结束后保持图表显示
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
    trained_agent = train(episodes=300000)
    print("\n训练完成！最终掌握的棋局状态数量:", len(trained_agent.q_table))
    play_human_vs_agent(trained_agent)
