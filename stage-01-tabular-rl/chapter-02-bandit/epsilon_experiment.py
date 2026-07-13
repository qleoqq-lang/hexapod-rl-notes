"""Compare epsilon-greedy agents on a stationary 10-armed bandit.


The experiment follows Chapter 2 of Sutton and Barto and the testbed used by
Shangtong Zhang's reinforcement-learning-an-introduction repository. It keeps
the seed, bandit instances, run count, and time steps fixed while changing only
epsilon.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_EPSILONS = (0.0, 0.01, 0.1, 0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--k-arms", type=int, default=10)
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(DEFAULT_EPSILONS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def choose_greedy_actions(
    q_estimates: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Choose uniformly among actions tied for the largest estimate."""
    best_values = q_estimates.max(axis=1, keepdims=True)
    is_best = q_estimates == best_values
    tie_breakers = rng.random(q_estimates.shape)
    tie_breakers[~is_best] = -1.0
    return tie_breakers.argmax(axis=1)


def simulate_epsilon(
    q_true: np.ndarray,
    epsilon: float,
    steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean reward and optimal-action rate for one epsilon."""
    runs, k_arms = q_true.shape
    rng = np.random.default_rng(seed)
    q_estimates = np.zeros((runs, k_arms), dtype=np.float64)
    action_counts = np.zeros((runs, k_arms), dtype=np.int32)
    optimal_actions = q_true.argmax(axis=1)
    run_indices = np.arange(runs)

    mean_rewards = np.zeros(steps, dtype=np.float64)
    optimal_action_rates = np.zeros(steps, dtype=np.float64)

    for step in range(steps):
        greedy_actions = choose_greedy_actions(q_estimates, rng)
        exploratory_actions = rng.integers(0, k_arms, size=runs)
        should_explore = rng.random(runs) < epsilon
        actions = np.where(should_explore, exploratory_actions, greedy_actions)

        rewards = q_true[run_indices, actions] + rng.normal(size=runs)
        action_counts[run_indices, actions] += 1
        counts = action_counts[run_indices, actions]
        estimates = q_estimates[run_indices, actions]
        q_estimates[run_indices, actions] += (rewards - estimates) / counts

        mean_rewards[step] = rewards.mean()
        optimal_action_rates[step] = np.mean(actions == optimal_actions)

    return mean_rewards, optimal_action_rates


def save_plot(
    x: np.ndarray,
    series: dict[float, np.ndarray],
    ylabel: str,
    title: str,
    output_path: Path,
    percentage: bool = False,
) -> None:
    plt.figure(figsize=(10, 6))
    for epsilon, values in series.items():
        plotted_values = values * 100 if percentage else values
        plt.plot(x, plotted_values, label=f"epsilon = {epsilon:g}")
    plt.xlabel("Time step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_parameter_table(
    output_path: Path,
    epsilons: list[float],
    seed: int,
    runs: int,
    steps: int,
    k_arms: int,
) -> None:
    fieldnames = [
        "epsilon",
        "seed",
        "runs",
        "time_steps",
        "k_arms",
        "true_action_value_distribution",
        "reward_distribution",
        "value_update",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for epsilon in epsilons:
            writer.writerow(
                {
                    "epsilon": epsilon,
                    "seed": seed,
                    "runs": runs,
                    "time_steps": steps,
                    "k_arms": k_arms,
                    "true_action_value_distribution": "Normal(0, 1)",
                    "reward_distribution": "Normal(q_true[action], 1)",
                    "value_update": "sample average",
                }
            )


def save_summary(
    output_path: Path,
    rewards: dict[float, np.ndarray],
    optimal_rates: dict[float, np.ndarray],
) -> None:
    fieldnames = [
        "epsilon",
        "early_mean_reward_steps_1_100",
        "late_mean_reward_last_100_steps",
        "early_optimal_action_rate_steps_1_100",
        "late_optimal_action_rate_last_100_steps",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for epsilon in rewards:
            early_window = min(100, len(rewards[epsilon]))
            writer.writerow(
                {
                    "epsilon": epsilon,
                    "early_mean_reward_steps_1_100": f"{rewards[epsilon][:early_window].mean():.6f}",
                    "late_mean_reward_last_100_steps": f"{rewards[epsilon][-early_window:].mean():.6f}",
                    "early_optimal_action_rate_steps_1_100": f"{optimal_rates[epsilon][:early_window].mean():.6f}",
                    "late_optimal_action_rate_last_100_steps": f"{optimal_rates[epsilon][-early_window:].mean():.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    if args.runs <= 0 or args.steps <= 0 or args.k_arms <= 1:
        raise ValueError("runs and steps must be positive; k-arms must exceed 1")
    if any(epsilon < 0 or epsilon > 1 for epsilon in args.epsilons):
        raise ValueError("each epsilon must be between 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment_rng = np.random.default_rng(args.seed)
    q_true = environment_rng.normal(size=(args.runs, args.k_arms))

    rewards: dict[float, np.ndarray] = {}
    optimal_rates: dict[float, np.ndarray] = {}
    for index, epsilon in enumerate(args.epsilons):
        print(f"Running epsilon={epsilon:g} ...")
        reward, optimal_rate = simulate_epsilon(
            q_true=q_true,
            epsilon=epsilon,
            steps=args.steps,
            seed=args.seed + 10_000 + index,
        )
        rewards[epsilon] = reward
        optimal_rates[epsilon] = optimal_rate

    x = np.arange(1, args.steps + 1)
    save_plot(
        x=x,
        series=rewards,
        ylabel="Average reward",
        title=f"Epsilon-greedy: average reward ({args.runs} runs, seed={args.seed})",
        output_path=args.output_dir / "epsilon-average-reward.png",
    )
    save_plot(
        x=x,
        series=optimal_rates,
        ylabel="Optimal action rate (%)",
        title=f"Epsilon-greedy: optimal action rate ({args.runs} runs, seed={args.seed})",
        output_path=args.output_dir / "epsilon-optimal-action-rate.png",
        percentage=True,
    )
    save_parameter_table(
        output_path=args.output_dir / "parameters.csv",
        epsilons=args.epsilons,
        seed=args.seed,
        runs=args.runs,
        steps=args.steps,
        k_arms=args.k_arms,
    )
    save_summary(
        output_path=args.output_dir / "summary.csv",
        rewards=rewards,
        optimal_rates=optimal_rates,
    )

    config = {
        "seed": args.seed,
        "runs": args.runs,
        "time_steps": args.steps,
        "k_arms": args.k_arms,
        "epsilons": args.epsilons,
        "comparison_rule": "Only epsilon changes; all methods share q_true testbeds.",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()

