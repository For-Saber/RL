
import numpy as np

class QAgent:
    def __init__(self):
        self.q_table = {}  # 状态-动作价值表 [^2]
        self.alpha = 0.1   # 学习率（更新步长）
        self.gamma = 0.9   # 未来奖励折扣系数 [^1]

    def get_state_key(self, board):
        """将棋盘状态转化为字符串键"""
        return ''.join(board)  # 例如['X','O',' '...]→"XO ..."[^3]

    def choose_action(self, state_key, valid_actions):
        """ε-greedy策略选择动作"""
        if np.random.rand() < 0.1 or state_key not in self.q_table:
            return np.random.choice(valid_actions)  # 探索 [^5]
        else:
            q_values = self.q_table[state_key]
            return valid_actions[np.argmax([q_values[a] for a in valid_actions])]  # 利用

def check_win(board, player):
        # 检查行
    for i in range(0, 9, 3):
        if board[i] == board[i + 1] == board[i + 2] == player:
                return True
    # 检查列
    for i in range(3):
        if board[i] == board[i + 3] == board[i + 6] == player:
                return True
    # 检查对角线
    if board[0] == board[4] == board[8] == player or board[2] == board[4] == board[6] == player:
        return True
    return False



def train():
    agent = QAgent()
    for episode in range(5000):  # 训练5000局
        board = [' '] * 9
        done = False
        while not done:
            state = agent.get_state_key(board)
            valid_actions = [i for i, c in enumerate(board) if c == ' ']
            if not valid_actions:
                break

            # 智能体行动
            action = agent.choose_action(state, valid_actions)
            row, col = divmod(action, 3)

            # 环境反馈（假设对手随机落子）
            board[action] = 'X'
            reward = 1 if check_win(board, 'X') else 0  # 获胜奖励 [^6]
            done = reward == 1 or ' ' not in board

            # Q值更新（你只需观察TD误差计算）
            if state not in agent.q_table:
                agent.q_table[state] = [0.0] * 9
            old_q = agent.q_table[state][action]

            next_state = agent.get_state_key(board)
            max_next_q = max(agent.q_table[next_state]) if next_state in agent.q_table else 0
            agent.q_table[state][action] = old_q + agent.alpha * (reward + agent.gamma * max_next_q - old_q)

def play_vs_agent(agent):
    board = [' ']*9
    for _ in range(4):
        action = agent.choose_action(agent.get_state_key(board), [i for i,c in enumerate(board) if c==' '])
        board[action] = 'X'
        if check_win(board, 'X'): return 'Agent Wins'
        # 玩家手动输入位置（示例）
        player_move = int(input("Enter position (0-8):"))
        board[player_move] = 'O'
        if check_win(board, 'O'): return 'Player Wins'
    return 'Draw'
