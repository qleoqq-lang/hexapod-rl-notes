# 学习路线与交付标准

这份路线图服务于六足机器人大创项目。学习顺序按“先理解决策问题，再学习函数逼近，最后进入机器人连续控制”组织。

## Stage 1：表格型强化学习基础

建议周期：2 周

- [x] 跑通 Chapter 2 原始代码并生成 Figure 2.1-2.6
- [ ] 完成 epsilon-greedy 参数对比实验
- [ ] 写出 `bandit_experiment.md`
- [ ] 跑通 Chapter 4 policy evaluation / value iteration
- [ ] 比较不同 gamma 下的 value table
- [ ] 写出 `gridworld_value_iteration.md`
- [ ] 跑通 Chapter 6 Sarsa / Q-learning / Cliff Walking
- [ ] 自己绘制 reward curve
- [ ] 写出 `sarsa_vs_qlearning.md`

达标标准：

- 能用自己的话解释 state、action、reward、policy、value 和 Q value。
- 能说明 epsilon-greedy 为什么同时包含探索和利用。
- 能解释 gamma 对短期奖励与长期奖励权衡的影响。
- 能从更新目标和实际路径解释 Sarsa 与 Q-learning 的差异。
- 每个结论都有对应代码、参数或图表作为证据。

## Stage 2：深度强化学习基础

在 Stage 1 达标后开始，重点包括 DQN、经验回放、目标网络和神经网络函数逼近。具体交付物将在进入该阶段时补充，避免路线图先于真实学习进度膨胀。

## Stage 3：连续控制

重点包括 policy gradient、Actor-Critic、PPO 和 SAC，并开始把 observation space、action space 与 reward function 映射到六足机器人任务。

## Stage 4：六足机器人应用

实验代码和项目结果进入 [hexapod-rl](https://github.com/qleoqq-lang/hexapod-rl)，本仓库继续记录算法选择、参数分析和失败实验。

