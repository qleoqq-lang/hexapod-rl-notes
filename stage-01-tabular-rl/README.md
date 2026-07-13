# Stage 1：表格型强化学习基础

这一阶段不使用深度学习，目标是先理解 agent 如何通过交互学习决策，以及不同更新规则为什么会产生不同策略。

## 章节导航

| 章节 | 主题 | 必须掌握 | 当前入口 |
|---|---|---|---|
| Chapter 2 | Multi-Armed Bandit | exploration / exploitation、epsilon-greedy、action value | [进入笔记](chapter-02-bandit/README.md) |
| Chapter 4 | Dynamic Programming | policy evaluation、Bellman equation、value iteration | [进入笔记](chapter-04-dynamic-programming/README.md) |
| Chapter 6 | TD Learning | TD error、Sarsa、Q-learning、on-policy / off-policy | [进入笔记](chapter-06-td-learning/README.md) |

三章的中文背景讲义集中放在 [Chapter 2/4/6 学习讲义](reference/chapter-02-04-06-study-guide.md)，用于第一次学习和后续查概念。每章目录中的 README 用于记录自己的实验与结论。

## 推荐顺序

```text
先读讲义中的对应章节
        ↓
只读懂原仓库的环境、动作选择、更新和实验入口
        ↓
原参数跑通一次，确认能够复现
        ↓
一次只修改一个参数
        ↓
保存图表并填写实验记录
        ↓
写出与六足机器人任务的联系
```

## 阶段交付物

- `chapter-02-bandit/bandit-experiment.md`
- `chapter-04-dynamic-programming/gridworld-value-iteration.md`
- `chapter-06-td-learning/sarsa-vs-qlearning.md`

文件尚未出现表示实验还没有完成。先做实验，再写结论。

