# 📘 hexapod-rl-notes — 六足机器人强化学习笔记与实验

[![Learning](https://img.shields.io/badge/Status-Learning-informational)](#-学习进度)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![Project](https://img.shields.io/badge/Project-hexapod--rl-2ea44f)](https://github.com/qleoqq-lang/hexapod-rl)

> **Learning notes and reproducible experiments for reinforcement learning in hexapod robotics.**
> 记录我为六足机器人大创项目学习强化学习的过程，包括概念理解、代码阅读、参数实验、结果分析，以及这些知识与机器人任务的联系。

---

## 📋 仓库目标

| 目标 | 这个仓库如何实现 |
|:---:|---|
| 后续复习 | 用章节导航、概念总结和实验结论快速找回知识 |
| 履历记录 | 保留可复现的代码、参数、图表和个人分析 |
| 帮助他人 | 说明学习顺序、常见困惑和实验观察，降低入门门槛 |

本仓库记录的是学习与验证过程；真正的六足机器人仿真、训练和控制代码放在 [hexapod-rl](https://github.com/qleoqq-lang/hexapod-rl)。

---

## 🧭 学习路线

| 阶段 | 内容 | 交付物 | 状态 |
|:---:|---|---|:---:|
| Stage 1 | 表格型强化学习基础 | Bandit、Value Iteration、Sarsa vs Q-learning | 🚧 进行中 |
| Stage 2 | 深度强化学习基础 | DQN、经验回放、目标网络 | ⏳ 计划中 |
| Stage 3 | 连续控制 | PPO、SAC、奖励设计 | ⏳ 计划中 |
| Stage 4 | 六足机器人应用 | 仿真训练、评估、Sim-to-Real | ⏳ 计划中 |

当前阶段详见 [Stage 1 总览](stage-01-tabular-rl/README.md)。

---

## 📈 学习进度

| 章节 | 核心问题 | 当前成果 | 状态 |
|---|---|---|:---:|
| Chapter 2 | 如何平衡探索与利用？ | 已跑通原仓库代码并生成 Figure 2.1-2.6 | 🚧 整理实验结论 |
| Chapter 4 | 如何由模型计算状态价值和最优策略？ | 学习讲义已准备 | ⏳ 待实验 |
| Chapter 6 | Sarsa 与 Q-learning 为什么行为不同？ | 学习讲义已准备 | ⏳ 待实验 |

进度只记录已经留下笔记、代码或图表证据的内容，避免把“计划学习”写成“已经掌握”。

---

## 🏗️ 仓库结构

```text
hexapod-rl-notes/
├── README.md
├── ROADMAP.md
├── stage-01-tabular-rl/
│   ├── README.md
│   ├── chapter-02-bandit/
│   │   ├── README.md
│   │   ├── figures-guide.md
│   │   └── figures/
│   ├── chapter-04-dynamic-programming/
│   │   └── README.md
│   ├── chapter-06-td-learning/
│   │   └── README.md
│   └── reference/
│       └── chapter-02-04-06-study-guide.md
└── templates/
    └── experiment-note.md
```

---

## 📝 每篇笔记的记录标准

一篇可以作为学习证据的笔记至少回答六个问题：

1. 我想验证什么？
2. 环境中的 state、action、reward 和 policy 是什么？
3. 改动了哪个变量，哪些条件保持不变？
4. 图表说明了什么？
5. 结果是否符合预期，为什么？
6. 这个结论与六足机器人有什么联系？

新实验可直接复制 [实验记录模板](templates/experiment-note.md)。

---

## 🔗 项目关系

```text
强化学习基础与可复现实验（本仓库）
                  ↓
六足机器人环境、奖励设计与算法训练（hexapod-rl）
                  ↓
大创阶段成果、演示视频与实验报告
```

例如，Bandit 中的探索与利用会影响机器人训练时的动作探索；Chapter 4 的 value 帮助理解长期回报；Chapter 6 的 on-policy / off-policy 区别会继续出现在 PPO、SAC 等算法中。

---

## 📚 学习来源

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), Sutton & Barto, 2nd Edition
- [reinforcement-learning-an-introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction), Shangtong Zhang

仓库中的原始书本图表复现实验会注明来源；笔记、参数对比、结论和后续实验由我自行整理。不会把上游仓库的完整代码复制进来，只保留自己的实验改动与分析。

---

## 👤 作者

**Q_Leo** — Soochow University AI 2025

