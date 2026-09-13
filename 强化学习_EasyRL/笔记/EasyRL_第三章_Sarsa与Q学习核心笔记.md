# 《Easy RL：强化学习教程》第三章 时序差分控制：Sarsa 与 Q 学习 核心笔记

> 对应教材：EasyRL v1.0.6 第 3.4.1–3.4.3 节（教材 p.61–65，PDF 第 69–73 页）

---

## 0. 一句话总览

时序差分（TD）方法按用途分两类：

* **TD 预测**：给定一个策略，估计价值函数 $V(s)$；
* **TD 控制**：用时序差分框架估计动作价值 $Q(s,a)$（直接估计 Q 表格，得到 Q 表格后即可更新策略）。

本章的 Sarsa 与 Q 学习都是用 TD 框架做控制的两种代表算法：

| 算法 | 类型 | 节/页码 | 更新时使用的下一步动作 |
| :--- | :--- | :--- | :--- |
| **Sarsa** | 同策略（on-policy）TD 控制 | 3.4.1（p.61–62） | 下一步**实际会执行**的动作 $a_{t+1}$ |
| **Q 学习** | 异策略（off-policy）TD 控制 | 3.4.2（p.62–64） | 下一步 Q 值**最大**的动作 $\arg\max_a Q(s_{t+1},a)$ |

> **一句话精髓**：
> Sarsa 是"自己走一步、看一步"地学（同策略，求稳）；Q 学习是"不管实际怎么走、只认最优"地学（异策略，求快）。

---

## 1. Sarsa：同策略时序差分控制（3.4.1，p.61–62）

### 1.1 定位：从 TD 预测到 TD 控制

* 时序差分方法原本是"给定一个策略，然后去估计它的价值函数"（预测 V）；
* Sarsa 所做的改变很简单（p.61）：**把原本 TD 方法更新 V 的过程，变成更新 Q**，即用时序差分框架来估计 Q 函数（控制）；
* 直接估计 Q 表格，得到 Q 表格后就可以更新策略。

### 1.2 核心更新公式（式 3.23 / 3.24，p.61）

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$

| 组成 | 含义 |
| :--- | :--- |
| **TD 目标**（时序差分目标） | $r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})$，是 $Q(s_t,a_t)$ 要逼近的目标值；其中 $Q(s_{t+1},a_{t+1})$ 近似 $G_{t+1}$ |
| **TD 误差** | 目标值 − 当前值，即 $r_{t+1} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)$ |
| **软更新** | $\alpha$ 类似学习率，每次只更新一点点，Q 值慢慢逼近真实的目标值 |

图 3.28（p.61）为时序差分单步更新的示意图：用"下一步"更新"这一步"。

### 1.3 为什么叫 Sarsa（p.61）

每次更新值函数需要知道五个量：当前**S**tate、当前**A**ction、**R**eward、下一步的**S**tate、下一步的**A**ction，即：

$$(s_t,\ a_t,\ r_{t+1},\ s_{t+1},\ a_{t+1})$$

因此得名 **Sarsa**。它走了一步之后，获取这五个值，就可以做一次更新。

### 1.4 完整应用流程（图 3.29 伪代码，p.62）

```text
算法参数：步长 α ∈ (0,1]，极小值 ε > 0
初始化：对所有 s ∈ S⁺、a ∈ A(s) 随机初始化 Q(s,a)，且 Q(终点, ·) = 0

对每一个回合循环：
    初始化状态 S
    使用从 Q 中衍生的策略（例如 ε-贪心策略）从 S 中选择动作 A
    对回合中的每一步循环：
        执行动作 A，观测奖励 R 与新状态 S′
        使用从 Q 中衍生的策略（例如 ε-贪心策略）从 S′ 中选择动作 A′   ← 同策略关键：A′ 是下一步实际会执行的动作
        Q(S,A) ← Q(S,A) + α[ R + γQ(S′,A′) − Q(S,A) ]              ← 软更新：目标值 − 当前值
        S ← S′；A ← A′
    直到 S 到达终点
```

* **单步更新**：每执行一个动作，就更新一次价值和策略；
* **同策略关键点**：更新所用的 $A'$ 就是下一步**真的要执行**的动作（由同一套 ε-贪心策略选出），而不是"理想中"的动作。

### 1.5 代码实现的两个方法（图 3.30，p.62）

智能体每与环境交互一次就可以学习一次，主要实现两个方法：

1. **根据 Q 表格选择动作，输出动作**；
2. **获取 $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$ 这几个值，更新 Q 表格**。

### 1.6 扩展：n 步 Sarsa 与 Sarsa(λ)（式 3.25–3.29，p.62）

Sarsa 属于单步更新（n = 1）：

$$Q_t^1 = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})$$

**n 步 Sarsa**：不进行单步更新，而是执行 n 步之后再更新价值与策略：

$$Q_t^n = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n Q(s_{t+n}, a_{t+n})$$

**Sarsa(λ)**：给 n 步回报加上资格迹衰减参数 λ 并求和：

$$Q_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} Q_t^n$$

更新策略为：$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ Q_t^\lambda - Q(s_t, a_t) \right]$

> 总之：Sarsa 和 Sarsa(λ) 的差别主要体现在**价值的更新**上。

---

## 2. Q 学习：异策略时序差分控制（3.4.2，p.62–64）

### 2.1 同策略 vs 异策略：两种策略的分工（p.62–63）

* **Sarsa（同策略）**：学习过程中只存在一种策略 π，用一种策略做动作的选取，也用同一种策略做优化（p.62）。它知道下一步的动作有可能跑到悬崖那边去，就会在优化策略时**尽可能离悬崖远一点**，保证下一步哪怕有随机动作，也还在安全区域内。
* **Q 学习（异策略）**：学习过程中有两种不同的策略（图 3.31，p.63）：

| 策略 | 符号 | 角色 | 通俗比喻 |
| :--- | :--- | :--- | :--- |
| **目标策略**（target policy） | $\pi$ | 需要去学习的策略，根据经验学习最优策略，**不需要与环境交互** | 在后方指挥战术的**军师** |
| **行为策略**（behavior policy） | $\mu$ | 探索环境的策略，大胆探索所有可能的轨迹、采集数据，把数据"喂"给目标策略学习 | 在前线探索的**战士** |

* 图 3.32（p.63）的比喻：环境是波涛汹涌的大海，学习策略太"胆小"无法直接交互，于是有探索策略——一个不畏风浪的**海盗**，非常激进地在环境中探索，把经验"写成稿子"喂给学习策略学习。
* **喂给目标策略的数据中并不需要 $a_{t+1}$**，而 Sarsa 是需要 $a_{t+1}$ 的（p.63）。

**异策略学习的好处**（p.63）：

1. 可以利用探索策略学到最佳的策略，**学习效率高**；
2. 可以学习其他智能体的动作，进行**模仿学习**（学习人或其他智能体产生的轨迹）；
3. 可以**重用旧的策略产生的轨迹**，探索过程需要很多计算资源，这样能节省资源。

### 2.2 Q 学习更新公式推导（p.64）

* 目标策略 π 直接在 Q 表格上使用贪心策略，取下一步能得到的所有状态（式 3.30）：

$$\pi(s_{t+1}) = \arg\max_{a'} Q(s_{t+1}, a')$$

* 行为策略 μ 可以是随机策略，但我们采取 ε-贪心策略，让行为策略基于 Q 表格逐渐改进、不至于是完全随机的。
* 构造 Q 学习目标：下一个动作都是通过 $\arg\max$ 操作选出来的（式 3.31）：

$$r_{t+1} + \gamma Q(s_{t+1}, A') = r_{t+1} + \gamma Q\left(s_{t+1},\ \arg\max_{a'} Q(s_{t+1}, a')\right) = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')$$

* 写成增量学习形式（式 3.32），时序差分目标变成 $r_{t+1} + \gamma \max_a Q(s_{t+1}, a)$：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$

### 2.3 Q 学习流程特点：不需要 A′（p.64–65）

* Q 学习更新 Q 表格时，用到的是 $Q(S', a)$ 对应的动作，它**不一定是下一个步骤会执行的实际动作**（因为实际执行的动作可能会探索）；
* Q 学习默认下一个动作不是通过行为策略选取的，而是**直接看 Q 表格、取最大化的值**，默认 A′ 为最佳策略选取的动作，所以**学习时不需要传入 A′（即 $a_{t+1}$）**；
* 图 3.34b（p.65）：Q 学习唯一与 Sarsa 不同的就是**不需要提前知道 $A_2$ 就能更新 $Q(S_1, A_1)$**；在一个回合的训练中，学习之前也不需要获取下一个动作 A′，只需要前面的 $(S, A, R, S')$；
* 行为策略可能有 0.1 的概率选择别的动作，但 Q 学习并不担心受探索的影响，它默认按照最佳的策略去优化目标策略，所以可以**更大胆地去寻找最优路径**，表现得比 Sarsa 大胆得多（p.64）。

### 2.4 与 Sarsa 的更新公式对比（图 3.33 / 3.34，p.64）

Sarsa 和 Q 学习的更新公式是一样的，区别只在**目标计算的部分**：

| 算法 | 目标计算（TD 目标） | 更新公式 |
| :--- | :--- | :--- |
| **Sarsa** | $r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})$（实际执行的动作） | $Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[ r_{t+1} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t) \right]$ |
| **Q 学习** | $r_{t+1} + \gamma \max_a Q(s_{t+1}, a)$（Q 值最大的动作） | $Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[ r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t,a_t) \right]$ |

* 图 3.34a：Sarsa 用自己的策略产生 $S, A, R, S', A'$ 这条轨迹，然后用 $Q(s_{t+1}, a_{t+1})$ 去更新原本的 Q 值；Q 学习并不需要知道实际上选择哪一个动作，它默认下一个动作就是 Q 值最大的那个动作。
* 补充史实（p.64）：**Q 学习算法被提出的时间更早，Sarsa 算法是 Q 学习算法的改进** [2]。

---

## 3. 同策略 vs 异策略的本质区别（3.4.3，p.65）

| 对比维度 | Sarsa（同策略） | Q 学习（异策略） |
| :--- | :--- | :--- |
| **策略数量** | 只用一种策略 π | 两种：目标策略 π + 行为策略 μ |
| **学习与交互** | 不仅用策略 π 学习，还用策略 π 与环境交互产生经验 | 分离目标策略与行为策略：行为策略探索采集数据，目标策略据此学习 |
| **目标策略** | ε-贪心（需兼顾探索，ε 值不断变小 → **策略不稳定**） | 贪心算法（直接根据行为策略采集到的数据采用最佳策略） |
| **行为策略** | 无（同一策略） | ε-贪心（可大胆探索） |
| **风格** | **保守"胆小"**：在悬崖行走问题中尽可能远离悬崖边，确保哪怕不小心探索了一点儿也还在安全区域 | **激进大胆**：希望每一步都获得最大的利益 |
| **更新公式** | 没有选取最大值的最大化操作 | 有 $\max$ 最大化操作 |
| **路径偏好** | 选择一条相对安全的迭代路线 | 更有可能探索到最佳策略（最短但可能贴着悬崖的最优路径） |

> **一句话记忆**：Sarsa 求稳（同策略、怕探索出事、路线保守），Q 学习求快（异策略、只认最优、大胆激进）。

图 3.35（p.65）为表格型方法总结图。

---

## 4. Q 学习解决悬崖寻路实验（3.5，p.65–68）

### 4.1 实验环境：CliffWalking（p.65–66）

* OpenAI Gym 开发的悬崖寻路（cliff walking）环境，是一个迷宫类问题（图 3.36）；
* **4×12 网格**：起点＝左下角编号 36，终点＝右下角编号 47，悬崖＝编号 37~46；
* 智能体每次可在上、下、左、右 4 个方向移动一步（动作 0/1/2/3），每移动一步得 **−1**；
* 三条规则（p.66）：
  1. 不能移出网格：想执行移出网格的动作时原地不动，但仍得 **−1**；
  2. 掉入悬崖：立即回到起点位置（36），得 **−100**（**注意：不结束回合，继续走**）；
  3. 到达终点：该回合结束，总奖励为各步奖励之和；
* 最少 13 步到达终点 → **最优每回合总奖励 = −13**，这是判断算法是否收敛的标准。

### 4.2 gymnasium 环境适配（实测 1.3.0）

教材代码基于 gym 0.x 旧接口，在 gymnasium 1.3.0 上需 4 处适配：

| 教材写法（gym 0.21） | gymnasium 1.3.0 写法 | 原因 |
| :--- | :--- | :--- |
| `gym.make('CliffWalking-v0')` | `gym.make('CliffWalking-v1')` | v0 已被弃用 |
| `env.seed(1)` | `np.random.seed(1)` | env.seed() 已移除 |
| `state = env.reset()` | `state, _ = env.reset()` | reset() 返回 (obs, info) |
| `next_state, reward, done, _ = env.step(a)` | `next_state, reward, terminated, truncated, _ = env.step(a)`；`done = terminated or truncated` | step() 返回 5 元组 |

> gymnasium 与 numpy 2.x 原生兼容，**不需要** np.bool8 兼容补丁（那是旧 gym 0.26 才需要的）。

### 4.3 代码结构与接口五步（p.66）

教材把训练抽象成 5 步接口：**① 初始化环境和智能体 → ② 每个回合智能体选动作 → ③ 环境反馈下一状态和奖励 → ④ 智能体更新策略 → ⑤ 多回合后收敛，保存模型、画图分析**。

| 代码单元 | 职责 | 对应教材 |
| :--- | :--- | :--- |
| 单元 1 导入+创建环境 | 初始化环境（48 状态 / 4 动作 / 起点 36） | 3.5.1（p.65–66） |
| 单元 2 QLearning 类 | 定义智能体：choose_action（选动作）+ update（更新） | 3.5.3（p.67–68） |
| 单元 3 训练主循环 | 跑 500 回合，记录 rewards 与滑动平均 | 3.5.2（p.66–67） |
| 单元 4 画训练曲线 | 可视化收敛情况 | 图 3.37（p.68） |
| 单元 5 测试 | 30 回合纯 argmax 验证 | 3.5.4（p.68） |

### 4.4 QLearning 类详解（p.67–68）

**__init__**：`Q_table = np.zeros((n_states, n_actions))`——48×4 全 0 的 Q 表格（行=状态，列=动作）。

**choose_action（ε-贪心，p.67）**：

```python
def choose_action(self, state):
    self.sample_count += 1
    self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
        math.exp(-1. * self.sample_count / self.epsilon_decay)
    if np.random.uniform(0, 1) > self.epsilon:
        return np.argmax(self.Q_table[state])   # 利用：取 Q 值最大动作
    return np.random.choice(self.n_actions)      # 探索：随机动作
```

* ε 随采样次数**指数递减**：$\varepsilon = \varepsilon_{end} + (\varepsilon_{start}-\varepsilon_{end}) \cdot \exp(-\text{sample\_count}/\varepsilon_{decay})$——前期多探索、后期多利用；
* 随机数 > ε → 利用（argmax）；否则 → 探索（随机）。

**update（式 3.33，p.68）**：

```python
def update(self, state, action, reward, next_state, done):
    Q_predict = self.Q_table[state][action]
    if done:
        Q_target = reward                        # 终止状态：目标值 = 本步奖励
    else:
        Q_target = reward + self.gamma * np.max(self.Q_table[next_state])
    self.Q_table[state][action] += self.lr * (Q_target - Q_predict)
```

* Q_predict＝当前 Q 值；Q_target＝TD 目标（$R + \gamma \max_a Q(S',a)$）；`+= lr × (目标−当前)`＝软更新；
* **终止状态时**获取不到下一个动作，直接令 `Q_target = reward`（p.68）。

### 4.5 训练主循环要点（p.66–67）

* 每回合：reset 回起点 → 循环（选动作 → step → update → 状态推进 → 累计奖励）→ 到终点（done）结束；
* **滑动平均**：`ma = 0.9 × 上一个ma + 0.1 × 本回合奖励`（p.67）——平滑单回合奖励的振荡，便于观察收敛趋势；
* **max_steps 防死循环保护**（教材代码没有）：因为掉崖不结束回合，若策略没学好可能永远到不了终点，故每回合加步数上限（如 200 步）强制结束。

### 4.6 测试与验证（p.68）

* 测试 20~50 回合（教材用 30 回合），**测试时直接 `np.argmax(Q_table[state])`、不探索**，专门检验学到的策略；
* 训练收敛后，30 个测试回合全部 = **−13**，说明智能体每次都走最优的 13 步到达终点（图 3.38 效果）；
* 若测试经常掉崖（奖励远差于 −13）：加大 train_eps 或调大 epsilon_decay。

### 4.7 完整可运行代码（gymnasium 1.3.0 适配版）

```python
# ===== 单元 1：导入 + 创建环境（教材 3.5.1，p.65-66） =====
import numpy as np
import math, warnings
import gymnasium as gym
warnings.filterwarnings('ignore')

env = gym.make('CliffWalking-v1')   # gymnasium 1.3.0 用 v1
np.random.seed(1)                   # 固定随机种子，可复现
n_states = env.observation_space.n  # 48
n_actions = env.action_space.n      # 4
print(f"状态数：{n_states}，动作数：{n_actions}")
state, _ = env.reset()              # reset() 返回 (obs, info)
print(f"初始状态：{state}")          # 预期 36

# ===== 单元 2：QLearning 类（教材 3.5.3，p.67-68） =====
class QLearning:
    def __init__(self, n_states, n_actions, cfg):
        self.n_actions = n_actions
        self.lr = cfg['lr']
        self.gamma = cfg['gamma']
        self.epsilon_start = cfg['epsilon_start']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay = cfg['epsilon_decay']
        self.sample_count = 0
        self.Q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay)
        if np.random.uniform(0, 1) > self.epsilon:
            return np.argmax(self.Q_table[state])
        return np.random.choice(self.n_actions)

    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[state][action]
        if done:
            Q_target = reward
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[next_state])
        self.Q_table[state][action] += self.lr * (Q_target - Q_predict)

# ===== 单元 3：训练主循环（教材 3.5.2，p.66-67） =====
cfg = {'lr': 0.1, 'gamma': 0.9, 'epsilon_start': 0.9, 'epsilon_end': 0.01, 'epsilon_decay': 200}
agent = QLearning(n_states, n_actions, cfg)

train_eps = 500
max_steps = 200        # 防死循环保护（教材原代码没有）
rewards, ma_rewards = [], []

for i_ep in range(train_eps):
    ep_reward = 0
    state, _ = env.reset()
    for step in range(max_steps):
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.update(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward
        if done:
            break
    rewards.append(ep_reward)
    ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1 if ma_rewards else ep_reward)

print("训练最后10回合奖励：", rewards[-10:])

# ===== 单元 4：画训练曲线（图 3.37，p.68） =====
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.4, label='每回合奖励')
    plt.plot(ma_rewards, label='滑动平均(0.9/0.1)')
    plt.axhline(-13, color='r', linestyle='--', label='理论最优 -13')
    plt.xlabel('回合数'); plt.ylabel('奖励'); plt.legend(); plt.show()
except ImportError:
    print('未安装 matplotlib，跳过画图')

# ===== 单元 5：测试 30 回合（教材 3.5.4，p.68） =====
test_eps = 30
test_rewards = []
for _ in range(test_eps):
    ep_reward = 0
    state, _ = env.reset()
    for step in range(max_steps):
        action = np.argmax(agent.Q_table[state])   # 测试纯利用，不探索
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        ep_reward += reward
        if terminated or truncated:
            break
    test_rewards.append(ep_reward)

print("测试30回合奖励：", test_rewards)
print("测试平均奖励：", round(float(np.mean(test_rewards)), 2), "（理论最优 -13）")
env.close()
```

> 实测结果（本机 gymnasium 1.3.0）：训练 500 回合后收敛，测试 30 回合全部 −13，平均 −13.0，与教材图 3.37/3.38 一致。

---

## 5. 页码溯源表

| 内容 | 教材页码 | PDF 页码 |
| :--- | :--- | :--- |
| 3.4.1 更新公式 (3.23)(3.24)、TD 目标/误差/软更新、图 3.28、名字由来 | p.61 | PDF 第 69 页 |
| 图 3.29 Sarsa 伪代码、n 步 Sarsa 与 Sarsa(λ)（式 3.25–3.29）、智能体两个方法（图 3.30 说明） | p.62 | PDF 第 70 页 |
| 3.4.2 开头（Sarsa 同策略、悬崖比喻） | p.62 | PDF 第 70 页 |
| 异策略概念：目标策略/行为策略、图 3.31、图 3.32、异策略好处 | p.63 | PDF 第 71 页 |
| Q 学习公式推导（式 3.30–3.32）、图 3.33 伪代码对比、图 3.34a、Q 学习大胆 | p.64 | PDF 第 72 页 |
| 图 3.34b（Q 学习流程拆解）、3.4.3 同策略与异策略的区别、图 3.35 | p.65 | PDF 第 73 页 |
| 3.5 环境规则与创建环境（图 3.36） | p.65–66 | PDF 第 74–75 页 |
| 3.5.2 接口五步、训练主循环、滑动平均 | p.66–67 | PDF 第 75–76 页 |
| 3.5.3 QLearning 类、choose_action、update 与式 3.33 | p.67–68 | PDF 第 76–77 页 |
| 3.5.4 结果分析、图 3.37/3.38、测试 30 回合 | p.68 | PDF 第 77 页 |

---

## 6. 常见问答速查（对应课后习题视角）

**Q：请描述基于 Sarsa 算法的智能体的学习过程。（习题 3-3）**

A：智能体初始化 Q 表格（终点 Q 值为 0）后，每回合从初始状态 S 出发，用 ε-贪心策略选动作 A 并执行；观测到 R、S′ 后再用 ε-贪心从 S′ 选出下一步实际会执行的动作 A′；随后用五元组 (S,A,R,S′,A′) 按式 (3.23) 做一次软更新，推进到 S←S′、A←A′，继续循环直到到达终点，再开启下一回合，直至 Q 表格收敛。

**Q：Q 学习算法和 Sarsa 算法的区别是什么？（习题 3-4）**

A：① Q 学习是异策略 TD 学习，Sarsa 是同策略 TD 学习；② Sarsa 更新 Q 表格时用到的 a′ 是获取下一个 Q 值时**一定会执行的动作**（ε-贪心采样出来的），Q 学习更新时用的是 Q 值**最大**对应的动作，不需要传入 A′；③ 更新公式中 Sarsa 的目标是 $r_{t+1}+\gamma Q(s_{t+1},a_{t+1})$，Q 学习的目标是 $r_{t+1}+\gamma\max_a Q(s_{t+1},a)$（多了 max 操作）；④ 表现上 Q 学习更大胆激进、可能找到更优路径，Sarsa 更保守安全。

---

*笔记整理基于《Easy RL：强化学习教程》v1.0.6 第 3.4–3.5 节，页码为教材印刷页码。*
