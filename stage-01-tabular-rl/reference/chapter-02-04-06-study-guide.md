# 强化学习入门讲义：Chapter 2、4、6

适用目标：没有 Sutton & Barto 教材，也能完成第一阶段“RL 基础，不碰深度学习”的学习任务。

这份讲义不是教材原文复刻，而是围绕你要跑的仓库 `reinforcement-learning-an-introduction`，用更工程化的方式解释第 2、4、6 章的核心思想。

你这一阶段的主线是：

```text
Chapter 2：先理解探索和利用
Chapter 4：再理解 value、policy、Bellman 更新
Chapter 6：最后理解 Sarsa、Q-learning 的行为差异
```

---

## 0. 先建立一张 RL 地图

强化学习研究的是一个 agent 如何在 environment 中不断试错，通过 reward 学会更好的 action 选择。

最小闭环是：

```text
state / observation
-> agent 根据 policy 选择 action
-> environment 执行动作
-> 得到 reward 和 next state
-> agent 更新自己的 value / policy
-> 重复
```

几个核心概念：

| 概念 | 含义 | 小例子 |
|---|---|---|
| state | 当前处境 | 机器人当前位置、网格坐标、游戏局面 |
| action | 可选动作 | 上下左右、选择第几个老虎机、输出关节目标 |
| reward | 环境给的即时反馈 | 到达目标 +10，掉坑 -100，每走一步 -1 |
| policy | 选择动作的规则 | 在某状态下选择哪个动作，或各动作的概率 |
| value | 长期好坏的估计 | 从这个状态出发，未来累计 reward 大概多少 |
| Q value | 状态-动作价值 | 在状态 s 选择动作 a 后，长期回报大概多少 |
| gamma | 折扣因子 | 越接近 1 越重视长期，越接近 0 越看重眼前 |
| alpha | 学习率 | 新经验对旧估计的影响程度 |
| epsilon | 探索概率 | epsilon-greedy 中随机试动作的概率 |

第一阶段最重要的是理解：RL 不是只最大化眼前 reward，而是在学习长期决策。

---

## 1. Chapter 2：Multi-Armed Bandit

### 1.1 这一章在讲什么

多臂老虎机是最简单的强化学习问题之一。

假设你面前有 10 台老虎机，每次只能选择其中一台。每台老虎机都有一个未知的平均奖励。有的机器长期收益高，有的长期收益低，但你一开始不知道哪台好。

每一步你要做一个选择：

```text
是选择当前看起来最好的机器？
还是冒险试试别的机器？
```

这就是强化学习中最经典的矛盾：

```text
exploitation：利用已有知识，选当前估计最好的动作
exploration：探索未知动作，收集更多信息
```

Chapter 2 的核心不是复杂算法，而是让你理解：

> 一个 agent 如果从不探索，可能会过早相信一个错误选择；如果一直探索，又无法稳定利用好选择。

### 1.2 Bandit 和完整 RL 的区别

Bandit 问题没有复杂的 state 转移。

普通 RL 是：

```text
state -> action -> reward -> next state
```

Bandit 更简单：

```text
action -> reward
```

每一步的环境状态基本不变，所以你只需要学习：

```text
哪个 action 的平均 reward 更高？
```

因此 Bandit 是理解 action value 的最好入口。

### 1.3 action value 是什么

对每个动作 a，我们关心它的真实价值：

```text
q*(a) = 选择动作 a 后的期望 reward
```

但真实价值 `q*(a)` agent 不知道，只能通过试验估计：

```text
Q(a) = agent 当前估计的动作价值
```

例如：

```text
动作 0：试了 20 次，平均 reward 约 0.8
动作 1：试了 5 次，平均 reward 约 1.1
动作 2：试了 1 次，reward 是 2.0
```

这时候动作 2 看起来最好，但它只试了一次，可能只是运气好。这就是为什么探索和估计不确定性很重要。

### 1.4 epsilon-greedy

epsilon-greedy 是最基础的探索策略。

规则是：

```text
以 1 - epsilon 的概率，选择当前 Q(a) 最大的动作
以 epsilon 的概率，随机选择一个动作
```

如果 `epsilon = 0`：

```text
永远选当前看起来最好的动作
```

优点是短期 reward 可能比较高，缺点是容易陷入早期误判。

如果 `epsilon = 0.1`：

```text
90% 时间利用当前最优动作
10% 时间随机探索
```

它既能利用已有知识，也能继续发现可能更好的动作。

如果 `epsilon` 太大：

```text
探索太多，长期 reward 会被随机动作拖低
```

所以 epsilon 是一种探索和利用之间的折中。

### 1.5 动作价值怎么更新

最直观的方法是样本平均：

```text
Q_n(a) = 到目前为止，动作 a 得到的所有 reward 的平均值
```

如果环境是稳定的，这个方法很自然。

但代码里更常见的是增量更新：

```text
新估计 = 旧估计 + alpha * (新 reward - 旧估计)
```

也就是：

```text
Q(a) <- Q(a) + alpha * [R - Q(a)]
```

括号里的：

```text
R - Q(a)
```

可以理解为预测误差：

```text
实际 reward 比我预期高多少或低多少？
```

`alpha` 控制更新幅度。

`alpha` 大：

```text
学得快，但曲线可能抖
```

`alpha` 小：

```text
学得慢，但估计更平滑
```

### 1.6 代码里应该看哪里

对应仓库文件通常是：

```text
chapter02/ten_armed_testbed.py
```

重点看：

```text
Bandit 类
act()
step(action)
simulate(...)
figure_2_2()
```

你要把代码变量和概念对起来：

| 代码变量 | 概念 |
|---|---|
| `q_true` | 每个动作的真实平均 reward，agent 不应该直接知道 |
| `q_estimation` | agent 对每个动作价值的估计 Q(a) |
| `epsilon` | 随机探索概率 |
| `step_size` | 学习率 alpha |
| `action_count` | 每个动作被选择了多少次 |
| `best_action` | 真实最优动作，用于统计是否选对 |

### 1.7 你应该做的实验

第一组：改 epsilon

```text
epsilon = 0
epsilon = 0.01
epsilon = 0.1
epsilon = 0.5
```

观察：

```text
平均 reward 曲线
最优动作选择比例曲线
```

你应该能解释：

```text
epsilon = 0 为什么可能一开始错了就一直错？
epsilon = 0.1 为什么长期通常更好？
epsilon 太大为什么最终 reward 上不去？
```

第二组：改 alpha

```text
alpha = 0.01
alpha = 0.1
alpha = 0.5
```

观察：

```text
学习速度
曲线波动程度
最终稳定水平
```

### 1.8 Chapter 2 你要能回答

1. 什么是 exploration 和 exploitation？
2. epsilon-greedy 为什么既探索又利用？
3. 为什么 epsilon = 0 不一定最好？
4. Q(a) 是真实值还是估计值？
5. alpha 变大，曲线可能出现什么变化？

---

## 2. Chapter 4：Dynamic Programming

### 2.1 这一章在讲什么

Chapter 4 进入真正的 MDP，也就是 Markov Decision Process。

相比 Bandit，MDP 多了 state 和 state transition：

```text
state -> action -> reward -> next state
```

这时你不再只是问：

```text
哪个 action 平均 reward 高？
```

而是问：

```text
在这个 state 下，选择什么 action，长期结果更好？
```

Dynamic Programming，动态规划，解决的是一种比较理想的情况：

```text
环境模型已知
```

也就是说，你知道：

```text
在 state s 执行动作 a，会以什么概率到达 next state s'
会得到什么 reward
```

有了完整模型，就可以通过 Bellman 更新计算 value 和最优策略。

### 2.2 policy evaluation

policy evaluation 的问题是：

```text
给定一个 policy，计算每个 state 的 value。
```

注意，这里 policy 是固定的。你不是在寻找最优策略，而是在评价一个已有策略有多好。

例如 GridWorld 里，如果策略是：

```text
每个方向都以 0.25 概率随机选择
```

那么 policy evaluation 要回答：

```text
从每个格子出发，如果一直随机走，长期累计 reward 期望是多少？
```

如果每走一步 reward = -1，那么离终点越远，value 往往越低。

### 2.3 value function

状态价值函数：

```text
V_pi(s)
```

意思是：

```text
从 state s 出发，之后一直按照 policy pi 行动，未来累计 reward 的期望。
```

这里的关键是“未来累计 reward”，不是当前一步 reward。

如果使用折扣因子 gamma：

```text
return = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ...
```

gamma 越大，越重视长期。

gamma 越小，越重视眼前。

### 2.4 Bellman 更新的直觉

Bellman 思想可以用一句话理解：

> 一个状态的价值，等于当前一步 reward 加上下一个状态价值的折扣。

简化写法：

```text
V(s) = reward + gamma * V(next_state)
```

如果 action 或 next state 有随机性，就对所有可能情况求平均。

GridWorld 中，如果 policy 是上下左右各 0.25，那么：

```text
V(s) = 0.25 * 每个方向结果的加权和
```

这就是 policy evaluation 的核心。

### 2.5 policy improvement

policy evaluation 是：

```text
我先评价当前 policy。
```

policy improvement 是：

```text
既然我知道每个 state 的 value 了，那我能不能改进 policy？
```

改进方法是：

```text
在每个 state 里，看看哪个 action 会带来更高的 reward + gamma * V(next_state)
然后选择更好的 action
```

这就是从“评价策略”走向“改进策略”。

### 2.6 policy iteration

policy iteration 是两个步骤反复交替：

```text
1. policy evaluation：评价当前策略
2. policy improvement：根据 value 改进策略
```

直到策略不再变化。

直觉上：

```text
先看现在这套走法有多好
再根据评价结果换一套更好的走法
再评价
再改进
```

### 2.7 value iteration

value iteration 更直接：

```text
每次更新 V(s) 时，直接假设自己会选择最好的 action
```

简化理解：

```text
V(s) <- max_a [reward + gamma * V(next_state)]
```

它不像 policy iteration 那样严格分成完整的 evaluation 和 improvement，而是边估值边朝最优方向推。

### 2.8 gamma 的影响

gamma 是折扣因子。

如果 `gamma` 接近 0：

```text
agent 更短视，更关心眼前 reward。
远处的奖励或惩罚影响很小。
```

如果 `gamma` 接近 1：

```text
agent 更长远，更关心未来累计结果。
远处的目标、风险、惩罚都会更重要。
```

对机器人任务来说，这个概念非常关键。

例如六足机器人：

```text
短期：迈大步可能立刻速度更快
长期：迈太猛可能失稳、摔倒、能耗变高
```

gamma 大时，策略更可能考虑后果。

gamma 小时，策略更容易追求眼前收益。

### 2.9 代码里应该看哪里

对应仓库文件通常有：

```text
chapter04/grid_world.py
chapter04/gamblers_problem.py
chapter04/car_rental.py
```

你第一阶段优先看：

```text
chapter04/grid_world.py
```

重点看：

```text
step(state, action)
compute_state_value(...)
figure_4_1()
```

你要把代码变量和概念对起来：

| 代码变量 | 概念 |
|---|---|
| `WORLD_SIZE` | 网格大小 |
| `ACTIONS` | 可选动作 |
| `ACTION_PROB` | 随机策略下各动作概率 |
| `discount` | gamma |
| `value` | 每个 state 的 V(s) |
| `new_value` | 一轮 Bellman 更新后的 value |

### 2.10 你应该做的实验

第一组：改 gamma

```text
discount = 1.0
discount = 0.9
discount = 0.5
```

观察：

```text
不同 state 的 value 绝对值如何变化？
离终点远的格子变化是否更明显？
```

第二组：改 reward

例如把每步 reward 从 `-1` 改成 `-0.1`。

观察：

```text
value table 整体尺度怎么变？
策略偏好是否变了？
```

第三组：改终点位置或障碍

这一步不一定在原代码里直接支持，但你可以作为理解练习：

```text
如果终点换位置，哪些 state 的 value 会提高？
如果某些格子惩罚很大，最优路径会不会绕开？
```

### 2.11 Chapter 4 你要能回答

1. policy evaluation 在评价什么？
2. V(s) 为什么不是当前 reward？
3. Bellman 更新的直觉是什么？
4. policy iteration 和 value iteration 有什么区别？
5. gamma 变大或变小会造成什么行为变化？

---

## 3. Chapter 6：Temporal-Difference Learning

### 3.1 这一章在讲什么

Chapter 6 是第一阶段最重要的一章。

它解决的问题是：

```text
如果环境模型未知，agent 能不能边行动边学习？
```

Chapter 4 的动态规划需要知道完整环境模型。但现实中，尤其是机器人任务里，我们通常不知道精确转移模型。

你不一定知道：

```text
这个动作之后机器人会精确到达什么状态
接触、滑移、扰动会怎样发生
```

所以我们需要从实际经验中学习：

```text
(state, action, reward, next_state)
```

这就是 TD learning 的入口。

### 3.2 TD 和 Monte Carlo 的区别

Monte Carlo 方法通常要等一个 episode 结束后，才知道完整 return。

例如：

```text
从起点走到终点，整局结束后，再回头更新之前经过的状态。
```

TD 方法不需要等整局结束。

它可以每走一步就更新：

```text
当前估计 <- 当前估计 + alpha * [一步 reward + gamma * 下一状态估计 - 当前估计]
```

核心优势：

```text
可以在线学习
可以边走边改
不需要等 episode 完整结束
```

### 3.3 TD error

TD 更新中最重要的量是 TD error：

```text
TD error = R + gamma * V(S') - V(S)
```

含义是：

```text
我原来以为 S 的价值是 V(S)
现在看到一步 reward 和下一个状态 S' 后
新的目标是 R + gamma * V(S')
两者差多少？
```

然后用 alpha 做小步更新：

```text
V(S) <- V(S) + alpha * TD error
```

这个结构在后面 DQN、Actor-Critic、PPO 的 critic 里都会反复出现。

### 3.4 从 V(s) 到 Q(s, a)

如果只是学 V(s)，你知道每个状态好不好，但还不一定知道该选哪个动作。

控制问题更常学习：

```text
Q(s, a)
```

意思是：

```text
在状态 s 选择动作 a，然后继续行动，未来累计 reward 大概是多少。
```

有了 Q(s, a)，选择动作就很简单：

```text
在当前 state 下，选择 Q 最大的 action
```

当然，为了探索，训练时通常还是用 epsilon-greedy。

### 3.5 Sarsa

Sarsa 的名字来自一条经验：

```text
S, A, R, S', A'
```

也就是：

```text
当前状态 S
当前动作 A
得到奖励 R
下一个状态 S'
下一个动作 A'
```

Sarsa 更新：

```text
Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
```

关键点是：

```text
它使用实际会执行的 next action A' 来更新。
```

如果你的行为策略是 epsilon-greedy，那么 A' 也是 epsilon-greedy 选出来的，可能是贪心动作，也可能是随机探索动作。

因此 Sarsa 是 on-policy：

```text
它学习的是当前这套包含探索行为的 policy 的价值。
```

### 3.6 Q-learning

Q-learning 更新：

```text
Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]
```

关键点是：

```text
它不管下一步实际会选什么动作，而是假设下一步会选择 Q 最大的动作。
```

因此 Q-learning 是 off-policy：

```text
行为上可以用 epsilon-greedy 探索
但学习目标朝着 greedy optimal policy 靠近
```

### 3.7 Sarsa 和 Q-learning 为什么结果不同

这是你必须真正理解的问题。

区别不在于它们“代码差一点点”，而在于它们对未来行为的假设不同。

Sarsa：

```text
我知道自己之后还会 epsilon-greedy 探索。
既然我未来可能随机走错，那我现在最好保守一点。
```

Q-learning：

```text
我用下一状态的最大 Q 值更新。
我假设未来会选择最优动作。
所以我更倾向于学最短、最优但可能更冒险的路线。
```

在 Cliff Walking 里，这个差异非常明显。

悬崖环境大概是：

```text
S . . . . . . . . . . G
  C C C C C C C C C C
```

从 S 到 G，如果贴着悬崖走，路径最短。

但训练时有 epsilon 探索，agent 偶尔会随机动作。如果贴着悬崖走，一次随机向下就会掉下去，reward = -100。

所以：

```text
Sarsa 往往学到更安全、离悬崖远一点的路线。
Q-learning 往往学到贴着悬崖的最短路线。
```

这就是 on-policy 和 off-policy 差异的直观例子。

### 3.8 Cliff Walking 中 reward 的意义

典型设置：

```text
普通移动：reward = -1
掉下悬崖：reward = -100，并回到起点
到达终点：episode 结束
```

这会形成一个取舍：

```text
短路径：步数少，普通惩罚少，但靠近悬崖风险大
安全路径：步数多，普通惩罚多，但不容易掉悬崖
```

这就是强化学习里的短期和长期取舍。

### 3.9 alpha 的影响

alpha 是学习率。

如果 alpha 太大：

```text
一次经验对 Q 值影响很大
学习快，但可能震荡
```

如果 alpha 太小：

```text
每次更新很保守
学习慢，但曲线平稳
```

在 Cliff Walking 里，alpha 过大时，曲线可能明显抖动，因为掉悬崖的 -100 会强烈影响估计。

### 3.10 epsilon 的影响

epsilon 控制探索。

epsilon 大：

```text
更容易发现不同路线
但训练过程中也更容易随机掉悬崖
平均 reward 可能更差
```

epsilon 小：

```text
行为更稳定
但可能探索不足
```

特别注意：

```text
Q-learning 学出来的 greedy 路线可能很好
但训练过程中的 epsilon 探索仍然可能让它掉悬崖
```

这也是为什么训练曲线和最终 greedy policy 不一定完全一致。

### 3.11 gamma 的影响

gamma 大：

```text
更重视未来惩罚和未来目标
会把远处悬崖、终点等长期因素考虑进去
```

gamma 小：

```text
更短视
主要看眼前 reward
可能不充分考虑后续风险或远期收益
```

在每步 reward 都是 -1 的任务里，gamma 的变化会影响 agent 对“快点结束”和“规避风险”的相对估计。

### 3.12 代码里应该看哪里

对应仓库文件通常是：

```text
chapter06/cliff_walking.py
```

重点看：

```text
choose_action(state, q_value)
sarsa(q_value)
q_learning(q_value)
figure_6_4()
```

你要把代码变量和概念对起来：

| 代码变量 | 概念 |
|---|---|
| `EPSILON` | epsilon-greedy 探索概率 |
| `ALPHA` | 学习率 |
| `GAMMA` | 折扣因子 |
| `q_value` | Q(s, a) 表 |
| `choose_action` | 根据 epsilon-greedy 选择动作 |
| `sarsa` | on-policy TD control |
| `q_learning` | off-policy TD control |
| `rewards_sarsa` | Sarsa 每个 episode 的 reward |
| `rewards_q_learning` | Q-learning 每个 episode 的 reward |

### 3.13 你应该做的实验

第一组：对比 Sarsa 和 Q-learning

保持默认参数：

```text
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1
```

观察：

```text
训练曲线谁更稳定？
谁更容易受到掉悬崖惩罚影响？
最终路径有什么差异？
```

第二组：改 epsilon

```text
EPSILON = 0
EPSILON = 0.01
EPSILON = 0.1
EPSILON = 0.3
```

重点解释：

```text
探索增多后，Sarsa 是否更保守？
Q-learning 的训练 reward 是否变差？
```

第三组：改 alpha

```text
ALPHA = 0.1
ALPHA = 0.5
ALPHA = 0.9
```

重点解释：

```text
学习速度和曲线波动如何变化？
```

第四组：改 gamma

```text
GAMMA = 0.5
GAMMA = 0.9
GAMMA = 1.0
```

重点解释：

```text
agent 是否更看重长期结果？
路径选择是否变化？
```

### 3.14 Chapter 6 你要能回答

1. TD 方法为什么不需要等 episode 结束？
2. TD error 是什么意思？
3. Sarsa 为什么叫 on-policy？
4. Q-learning 为什么叫 off-policy？
5. 为什么 Cliff Walking 中 Sarsa 和 Q-learning 学到的行为不同？
6. 训练 reward 曲线和最终策略为什么可能不完全一致？

---

## 4. 三章之间的关系

你可以这样理解它们的递进：

```text
Chapter 2：我只需要知道哪个动作平均奖励高
Chapter 4：我知道环境模型，可以计算每个状态长期价值
Chapter 6：我不知道环境模型，但可以通过实际经验边走边学
```

从机器人角度看：

```text
Chapter 2：学会探索和利用，不要只相信当前最好动作
Chapter 4：理解长期价值，知道策略不是只看眼前 reward
Chapter 6：理解边交互边学习，为后续 DQN / PPO / SAC 打基础
```

---

## 5. 第一阶段学习顺序

建议顺序：

```text
1. 先读这份讲义的 Chapter 2 部分
2. 跑 chapter02/ten_armed_testbed.py
3. 改 epsilon，画 reward 曲线
4. 写 bandit_experiment.md

5. 读这份讲义的 Chapter 4 部分
6. 跑 chapter04/grid_world.py
7. 改 gamma，观察 value table
8. 写 gridworld_value_iteration.md

9. 读这份讲义的 Chapter 6 部分
10. 跑 chapter06/cliff_walking.py
11. 改 epsilon / alpha / gamma
12. 写 sarsa_vs_qlearning.md
```

---

## 6. 你最终要能用自己的话讲出来

### 6.1 一个最简单的 RL 过程

可以这样讲：

```text
agent 在某个 state 下，根据 policy 选择 action。
环境执行 action 后返回 reward 和 next state。
agent 根据 reward 和 next state 更新自己对 value 或 Q value 的估计。
经过很多次试错后，agent 会更倾向于选择长期 reward 更高的动作。
```

### 6.2 Q-learning 和 Sarsa 为什么不同

可以这样讲：

```text
Sarsa 用实际下一步会执行的动作更新，所以它会考虑当前探索策略带来的风险。
Q-learning 用下一状态中最大的 Q 值更新，所以它朝着贪心最优策略学习。
在 Cliff Walking 中，Sarsa 知道自己还会随机探索，因此倾向于远离悬崖；Q-learning 假设未来总能选最优动作，因此更容易学到贴着悬崖的最短路径。
```

### 6.3 epsilon-greedy 为什么既探索又利用

可以这样讲：

```text
epsilon-greedy 大多数时候选择当前估计最好的动作，这是利用；少数时候随机选择动作，这是探索。探索可以发现之前低估的动作，利用可以把已经学到的好动作转化为更高 reward。
```

### 6.4 gamma 变大或变小的影响

可以这样讲：

```text
gamma 越大，未来 reward 的影响越大，agent 更重视长期结果。
gamma 越小，未来 reward 被快速折扣，agent 更短视，更重视眼前 reward。
```

### 6.5 为什么短期和长期奖励之间有取舍

可以这样讲：

```text
有些动作眼前 reward 高，但会带来未来风险；有些动作眼前 reward 低，但长期更稳定。强化学习的目标通常是最大化累计 reward，所以策略需要在眼前收益和未来后果之间取舍。
```

---

## 7. 建议交付物模板

你可以按下面方式写三篇笔记。

### bandit_experiment.md

```markdown
# Bandit 实验

## 实验目标
比较不同 epsilon 对平均 reward 和最优动作选择比例的影响。

## 我理解的环境
- state：
- action：
- reward：
- policy：
- value：

## 参数设置
- epsilon：
- alpha：
- runs：
- time steps：

## 结果
插入 reward 曲线和 optimal action 曲线。

## 结论
epsilon 太小的问题：
epsilon 太大的问题：
我认为比较合适的 epsilon：
```

### gridworld_value_iteration.md

```markdown
# GridWorld Value 实验

## 实验目标
理解 policy evaluation / value iteration，以及 gamma 对 value 的影响。

## 我理解的环境
- state：
- action：
- reward：
- policy：
- value：

## 参数设置
- gamma：
- reward：
- terminal state：

## 结果
插入不同 gamma 下的 value table。

## 结论
gamma 变大：
gamma 变小：
value 为什么体现长期回报：
```

### sarsa_vs_qlearning.md

```markdown
# Sarsa vs Q-learning

## 实验目标
比较 Sarsa 和 Q-learning 在 Cliff Walking 中的行为差异。

## 我理解的环境
- state：
- action：
- reward：
- policy：
- Q value：

## 参数设置
- epsilon：
- alpha：
- gamma：
- episodes：

## 结果
插入 reward 曲线。

## 结论
Sarsa 的行为：
Q-learning 的行为：
为什么二者不同：
```

---

## 8. 判断自己是否学会

如果你能做到下面这些，就可以进入第二阶段：

```text
能解释 state、action、reward、policy、value、Q value
能跑出 Bandit 曲线
能解释 epsilon 的作用
能看懂 GridWorld 的 value table
能解释 gamma 的作用
能跑出 Cliff Walking 曲线
能说明 Sarsa 和 Q-learning 的差异
能写出三篇简短实验记录
```

如果还做不到，不要急着进入 DQN / PPO。因为深度强化学习只是把表格换成神经网络，基本问题仍然是这些。

