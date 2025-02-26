
import numpy as np
import matplotlib.pyplot as plt

# ========== 参数设置 ==========
num_arms = 2  # 老虎机臂数
epsilon = 0.1  # 探索概率
num_steps = 10000  # 总步数
true_means = [500, 550]  # 真实均值(左臂,右臂)
true_stds = [20, 40]  # 真实标准差

# ========== 初始化 ==========
Q = np.full(num_arms,998)  # 各臂的价值估计(初始化为0)
N = np.zeros(num_arms)  # 各臂被选择的次数
rewards = []  # 记录每一步的奖励
choices = []  # 记录每一步的选择

# ========== 主循环 ==========
for step in range(num_steps):
    # ε-Greedy策略选择动作
    if np.random.random() < epsilon:
        # 探索：随机选择臂
        action = np.random.randint(num_arms)
    else:
        # 利用：选择当前估值最高的臂
        action = np.argmax(Q)

    # 生成奖励（根据选择的臂）
    reward = np.random.normal(loc=true_means[action], scale=true_stds[action])

    # 更新价值估计（增量式计算平均值）
    N[action] += 1
    Q[action] = (reward + Q[action]) / 2

    # 记录数据
    rewards.append(reward)
    choices.append(action)

# ========== 结果可视化 ==========
plt.figure(figsize=(12, 5))

# 绘制平均奖励曲线
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(rewards) / (np.arange(num_steps) + 1))
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward Over Time')

# 绘制动作选择分布
plt.subplot(1, 2, 2)
plt.bar(['Arm 0', 'Arm 1'], [sum(np.array(choices) == 0), sum(np.array(choices) == 1)])
plt.ylabel('Selection Count')
plt.title('Arm Selection Distribution')

plt.tight_layout()
plt.show()



# ========== 结果分析 ==========
print(f"最终价值估计: Arm0={Q[0]:.1f}, Arm1={Q[1]:.1f}")
print(f"真实均值: Arm0={true_means[0]}, Arm1={true_means[1]}")
print(f"选择次数: Arm0={N[0]}, Arm1={N[1]}")