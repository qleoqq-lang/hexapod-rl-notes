# Chapter 2 图表分类导读

本文件用于整理从 `reinforcement-learning-an-introduction/images/` 生成并复制到本仓库的 Chapter 2 图表。

Chapter 2 的主题是 multi-armed bandit，核心问题是：

```text
在不知道哪个动作最好的情况下，agent 如何在探索和利用之间做取舍？
```

---

## 一、主线必看图

这些图直接对应第一阶段的达标目标，建议写进学习笔记和 GitHub 展示页。

| 图表 | 文件名 | 重点问题 | 建议用途 |
|---|---|---|---|
| Figure 2.1 | `figure_2_1.png` | 10 个动作的 reward 分布长什么样 | 作为 bandit 问题的直观背景图 |
| Figure 2.2 | `figure_2_2.png` | 不同 epsilon-greedy 策略的平均 reward 和最优动作比例 | Chapter 2 最重要的实验图 |

### Figure 2.1 怎么理解

这张图展示的是 10 个动作各自的 reward 分布。

每个动作不是固定给一个 reward，而是从一个概率分布中采样。某些动作的平均 reward 更高，但单次采样仍然可能波动。

它说明：

```text
agent 不能只凭一次 reward 就判断一个动作好坏。
```

### Figure 2.2 怎么理解

这张图比较：

```text
epsilon = 0
epsilon = 0.01
epsilon = 0.1
```

通常可以观察到：

```text
epsilon = 0：只利用，不探索，可能早期选错后一直错
epsilon = 0.01：探索较少，学习较慢
epsilon = 0.1：探索更多，长期更容易找到好动作
```

这张图是你解释 exploration / exploitation 的核心证据。

---

## 二、扩展理解图

这些图有助于加深 Chapter 2，但不是第一轮必须精读。

| 图表 | 文件名 | 主题 | 当前建议 |
|---|---|---|---|
| Figure 2.3 | `figure_2_3.png` | optimistic initial values，乐观初始值 | 理解“初始估计也能鼓励探索” |
| Figure 2.4 | `figure_2_4.png` | UCB，置信上界探索 | 暂时知道它是另一种探索策略即可 |
| Figure 2.5 | `figure_2_5.png` | gradient bandit，有无 baseline 的影响 | 可放到后面学 policy gradient 前再看 |
| Figure 2.6 | `figure_2_6.png` | 多种 bandit 方法的参数敏感性比较 | 适合作为扩展总结图 |

### Figure 2.3 怎么理解

乐观初始值的想法是：

```text
一开始故意把每个动作的估计值设得很高。
```

当 agent 试过某个动作后，如果 reward 没有想象中那么好，这个动作的估计值会下降。于是 agent 会转去尝试其他还没被试过的动作。

所以它即使 `epsilon = 0`，也能在早期产生探索效果。

### Figure 2.4 怎么理解

UCB 不只是看当前估计值高不高，还会考虑：

```text
这个动作被试得少不少？
```

如果一个动作试得少，它的不确定性更大，UCB 会给它额外加分，鼓励探索。

### Figure 2.5 怎么理解

gradient bandit 不直接估计每个动作的 reward 平均值，而是学习动作偏好，再通过概率选择动作。

baseline 的作用是降低更新波动。这个思想后面在 policy gradient、Actor-Critic、PPO 中还会出现。

### Figure 2.6 怎么理解

这张图横向比较多种 bandit 算法在不同参数下的效果。

第一阶段不需要把所有方法都掌握，只需要记住：

```text
强化学习算法对超参数敏感。
同一种算法，参数不同，曲线可能差很多。
```