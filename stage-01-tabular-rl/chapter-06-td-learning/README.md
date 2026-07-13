# Chapter 6：Temporal-Difference Learning

## 核心问题

不知道完整环境模型时，agent 如何一边交互、一边更新 value 或 Q value？Sarsa 与 Q-learning 为什么会学出不同的行为？

## 必须掌握

- TD Learning 用当前估计去更新当前估计，即 bootstrapping。
- TD error 衡量“实际一步结果 + 下一状态估计”与当前估计之间的差距。
- Sarsa 使用实际策略下一步选择的动作更新，是 on-policy 方法。
- Q-learning 使用下一状态中最大的 Q value 更新，是 off-policy 方法。
- Cliff Walking 中，Sarsa 会把探索失误的风险计入学习目标，通常更保守；Q-learning 学的是贪心目标策略，通常更贴近悬崖边的最短路线。

## 计划实验

1. 跑通 Sarsa、Q-learning 和 Cliff Walking。
2. 固定环境与随机种子，使用相同 epsilon、alpha、gamma 和 episodes。
3. 绘制每个 episode 的累计 reward 曲线。
4. 展示两种算法最终路径，并解释差异来自哪一个更新目标。
5. 分别修改 epsilon、alpha、gamma，一次只改一个变量。

完成后新建 `sarsa-vs-qlearning.md`，并使用 [实验记录模板](../../templates/experiment-note.md)。

## 与六足机器人的联系

机器人训练中的探索动作可能导致摔倒、碰撞或高能耗。算法是否把当前探索策略的风险计入更新，会影响学到的策略是更激进还是更稳健。Cliff Walking 是理解这一差异的最小实验。
