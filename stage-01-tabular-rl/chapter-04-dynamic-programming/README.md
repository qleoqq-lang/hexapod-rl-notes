# Chapter 4：Dynamic Programming

## 核心问题

已知环境的状态转移和奖励规则时，怎样计算一个策略的价值，并进一步找到最优策略？

## 必须掌握

- `V(s)` 表示从状态 `s` 出发、之后遵循某个策略时的期望累计回报。
- Policy Evaluation 固定策略，反复使用 Bellman expectation update 估计状态价值。
- Policy Improvement 根据当前 value 选择更好的动作。
- Value Iteration 把策略改进合并到 value 更新中。
- gamma 越大越重视远期回报，越小越重视近期回报。

## 计划实验

1. 跑通 GridWorld 的 policy evaluation / value iteration。
2. 保存收敛后的 value table 和 policy。
3. 比较至少三组 gamma，例如 0.5、0.9、0.99。
4. 解释 terminal state、step reward 和 gamma 如何共同影响路径选择。

完成后新建 `gridworld-value-iteration.md`，并使用 [实验记录模板](../../templates/experiment-note.md)。

## 与六足机器人的联系

机器人一次动作的价值不只由当前是否前进决定，还取决于它是否让机身保持稳定、是否为后续动作留下良好姿态，以及最终能否抵达目标。Value function 正是在表达这种长期后果。
