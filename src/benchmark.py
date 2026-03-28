"""
Benchmark & Analysis Module.

Runs systematic experiments comparing pathfinding algorithms and RL agents
across different grid sizes, obstacle densities, and maze types.
Produces DataFrames and summary statistics.
"""

from __future__ import annotations

import time
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from .environment import GridWorld, generate_maze_dfs, generate_maze_prim
from .pathfinding import ALL_ALGORITHMS, SearchResult, run_all
from .rl_agent import QLearningAgent, SARSAAgent, TrainingResult


def benchmark_pathfinding(
    sizes: List[int] = None,
    obstacle_ratios: List[float] = None,
    n_trials: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run pathfinding algorithms across varying grid sizes and obstacle densities.
    Returns a DataFrame with performance metrics.
    """
    if sizes is None:
        sizes = [11, 21, 31, 51]
    if obstacle_ratios is None:
        obstacle_ratios = [0.1, 0.2, 0.3]

    records = []
    for size in sizes:
        for ratio in obstacle_ratios:
            for trial in range(n_trials):
                trial_seed = seed + trial + int(size * 100 + ratio * 1000)
                env = GridWorld(size, size)
                env.add_random_obstacles(ratio=ratio, seed=trial_seed)
                env.add_random_swamps(ratio=0.05, seed=trial_seed + 1)

                results = run_all(env)
                for result in results:
                    records.append({
                        "grid_size": size,
                        "obstacle_ratio": ratio,
                        "trial": trial,
                        "algorithm": result.algorithm,
                        "found": result.found,
                        "path_length": result.path_length,
                        "path_cost": result.path_cost,
                        "nodes_explored": result.nodes_explored,
                        "time_ms": result.time_seconds * 1000,
                    })

    return pd.DataFrame(records)


def benchmark_mazes(
    sizes: List[int] = None,
    n_trials: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Run pathfinding on generated mazes (DFS and Prim's)."""
    if sizes is None:
        sizes = [11, 21, 31, 51]

    records = []
    generators = {"DFS Maze": generate_maze_dfs, "Prim Maze": generate_maze_prim}

    for size in sizes:
        for maze_name, gen_fn in generators.items():
            for trial in range(n_trials):
                trial_seed = seed + trial
                env = gen_fn(size, size, seed=trial_seed)
                results = run_all(env)
                for result in results:
                    records.append({
                        "grid_size": size,
                        "maze_type": maze_name,
                        "trial": trial,
                        "algorithm": result.algorithm,
                        "found": result.found,
                        "path_length": result.path_length,
                        "path_cost": result.path_cost,
                        "nodes_explored": result.nodes_explored,
                        "time_ms": result.time_seconds * 1000,
                    })

    return pd.DataFrame(records)


def benchmark_rl(
    grid_size: int = 11,
    obstacle_ratio: float = 0.15,
    episodes: int = 1000,
    n_trials: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare Q-Learning and SARSA on the same environment."""
    records = []

    for trial in range(n_trials):
        trial_seed = seed + trial
        env = GridWorld(grid_size, grid_size)
        env.add_random_obstacles(ratio=obstacle_ratio, seed=trial_seed)

        for AgentClass, name in [(QLearningAgent, "Q-Learning"), (SARSAAgent, "SARSA")]:
            agent = AgentClass(env.copy(), seed=trial_seed)
            t0 = time.perf_counter()
            result = agent.train(episodes=episodes)
            elapsed = time.perf_counter() - t0

            final_reward = np.mean(result.episode_rewards[-100:]) if len(result.episode_rewards) >= 100 else np.mean(result.episode_rewards)
            reached_goal = result.policy_path[-1] == env.goal if result.policy_path else False

            records.append({
                "trial": trial,
                "algorithm": name,
                "episodes": episodes,
                "final_avg_reward": final_reward,
                "path_length": len(result.policy_path),
                "reached_goal": reached_goal,
                "converged_at": result.converged_at,
                "training_time_s": elapsed,
            })

    return pd.DataFrame(records)


def summary_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Compute summary statistics grouped by specified columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg_cols = [c for c in numeric_cols if c not in group_cols and c != "trial"]
    return df.groupby(group_cols)[agg_cols].agg(["mean", "std"]).round(3)
