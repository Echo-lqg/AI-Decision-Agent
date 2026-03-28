"""
CLI Entry Point — AI Decision Agent.

Usage:
    python main.py pathfinding [--size 15] [--seed 42] [--maze dfs]
    python main.py rl [--size 11] [--episodes 1000] [--seed 42]
    python main.py benchmark [--type pathfinding|maze|rl]
    python main.py demo
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment import GridWorld, generate_maze_dfs, generate_maze_prim
from src.pathfinding import run_all
from src.rl_agent import QLearningAgent, SARSAAgent
from src.visualizer import (
    plot_comparison, plot_exploration_heatmap,
    plot_training_curves, plot_value_map, plot_rl_path_on_grid,
)
from src.benchmark import benchmark_pathfinding, benchmark_mazes, benchmark_rl


def make_env(args) -> GridWorld:
    maze = getattr(args, "maze", None)
    size = args.size
    seed = args.seed

    if maze == "dfs":
        return generate_maze_dfs(size, size, seed=seed)
    elif maze == "prim":
        return generate_maze_prim(size, size, seed=seed)
    else:
        env = GridWorld(size, size)
        env.add_random_obstacles(ratio=0.2, seed=seed)
        env.add_random_swamps(ratio=0.05, seed=seed + 1)
        env.add_random_rewards(count=3, seed=seed + 2)
        return env


def cmd_pathfinding(args):
    env = make_env(args)
    print(f"[GridWorld] {env.rows}×{env.cols}  start={env.start}  goal={env.goal}")

    results = run_all(env)
    print(f"\n{'Algorithm':<12} {'Found':<7} {'Path':<6} {'Cost':<8} {'Explored':<10} {'Time (ms)'}")
    print("-" * 60)
    for r in results:
        print(f"{r.algorithm:<12} {'Yes' if r.found else 'No':<7} {r.path_length:<6} "
              f"{r.path_cost:<8.1f} {r.nodes_explored:<10} {r.time_seconds*1000:.3f}")

    fig = plot_comparison(env, results)
    fig.savefig("output_pathfinding_comparison.png", dpi=150, bbox_inches="tight")
    print("\n Saved: output_pathfinding_comparison.png")

    fig = plot_exploration_heatmap(env, results)
    fig.savefig("output_exploration_heatmap.png", dpi=150, bbox_inches="tight")
    print("\n Saved: output_exploration_heatmap.png")
    plt.close("all")


def cmd_rl(args):
    env = make_env(args)
    print(f"[GridWorld] {env.rows}×{env.cols}  Training {args.episodes} episodes...")

    training_results = []
    for AgentClass, name in [(QLearningAgent, "Q-Learning"), (SARSAAgent, "SARSA")]:
        print(f"\n  Training {name}...", end=" ", flush=True)
        agent = AgentClass(env.copy(), seed=args.seed)
        result = agent.train(episodes=args.episodes)
        training_results.append(result)

        reached = result.policy_path[-1] == env.goal if result.policy_path else False
        print(f"Done. Path={len(result.policy_path)} Reached={'Yes' if reached else 'No'} "
              f"Converged={result.converged_at or 'N/A'}")

    fig = plot_training_curves(training_results)
    fig.savefig("output_rl_training_curves.png", dpi=150, bbox_inches="tight")
    print("\n✅ Saved: output_rl_training_curves.png")

    for result in training_results:
        fig = plot_rl_path_on_grid(env, result)
        fname = f"output_rl_{result.algorithm.replace('-', '').lower()}_path.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {fname}")

        fig = plot_value_map(env, result.q_table, f"{result.algorithm} Value Map")
        fname = f"output_rl_{result.algorithm.replace('-', '').lower()}_valuemap.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {fname}")

    plt.close("all")


def cmd_benchmark(args):
    bench_type = args.type

    if bench_type == "pathfinding":
        print("Running pathfinding benchmark...")
        df = benchmark_pathfinding(seed=args.seed)
        print(df.to_string(index=False))
        df.to_csv("output_benchmark_pathfinding.csv", index=False)
        print("\n✅ Saved: output_benchmark_pathfinding.csv")

    elif bench_type == "maze":
        print("Running maze benchmark...")
        df = benchmark_mazes(seed=args.seed)
        print(df.to_string(index=False))
        df.to_csv("output_benchmark_maze.csv", index=False)
        print("\n✅ Saved: output_benchmark_maze.csv")

    elif bench_type == "rl":
        print("Running RL benchmark...")
        df = benchmark_rl(seed=args.seed)
        print(df.to_string(index=False))
        df.to_csv("output_benchmark_rl.csv", index=False)
        print("\n✅ Saved: output_benchmark_rl.csv")


def cmd_demo(args):
    print("=" * 60)
    print("  AI Decision Agent — Full Demo")
    print("=" * 60)

    # 1) Random grid pathfinding
    print("\n[1/4] Random Grid — Pathfinding Comparison")
    env = GridWorld(15, 15)
    env.add_random_obstacles(ratio=0.2, seed=42)
    env.add_random_swamps(ratio=0.05, seed=43)
    env.add_random_rewards(count=3, seed=44)
    results = run_all(env)
    for r in results:
        print(f"  {r.algorithm:<12} cost={r.path_cost:<8.1f} explored={r.nodes_explored}")
    fig = plot_comparison(env, results)
    fig.savefig("demo_1_pathfinding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) Maze pathfinding
    print("\n[2/4] DFS Maze — Pathfinding")
    maze = generate_maze_dfs(21, 21, seed=42)
    results = run_all(maze)
    for r in results:
        print(f"  {r.algorithm:<12} cost={r.path_cost:<8.1f} explored={r.nodes_explored}")
    fig = plot_comparison(maze, results)
    fig.savefig("demo_2_maze.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) RL training
    print("\n[3/4] RL Training (Q-Learning vs SARSA, 1000 episodes)")
    rl_env = GridWorld(11, 11)
    rl_env.add_random_obstacles(ratio=0.15, seed=42)
    rl_results = []
    for AgentClass, name in [(QLearningAgent, "Q-Learning"), (SARSAAgent, "SARSA")]:
        agent = AgentClass(rl_env.copy(), seed=42)
        result = agent.train(episodes=1000)
        rl_results.append(result)
        reached = result.policy_path[-1] == rl_env.goal if result.policy_path else False
        print(f"  {name:<12} path={len(result.policy_path)} reached={reached}")
    fig = plot_training_curves(rl_results)
    fig.savefig("demo_3_rl_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4) Value maps
    print("\n[4/4] Value Maps & Policy Visualization")
    for result in rl_results:
        fig = plot_value_map(rl_env, result.q_table, f"{result.algorithm} V(s)")
        fname = f"demo_4_{result.algorithm.replace('-', '').lower()}_value.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\n" + "=" * 60)
    print("  Demo complete! Check demo_*.png files.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="AI Decision Agent — Path Planning & Reinforcement Learning"
    )
    sub = parser.add_subparsers(dest="command")

    # pathfinding
    p1 = sub.add_parser("pathfinding", help="Run pathfinding algorithms")
    p1.add_argument("--size", type=int, default=15)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--maze", choices=["dfs", "prim"], default=None)

    # rl
    p2 = sub.add_parser("rl", help="Train RL agents")
    p2.add_argument("--size", type=int, default=11)
    p2.add_argument("--episodes", type=int, default=1000)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--maze", choices=["dfs", "prim"], default=None)

    # benchmark
    p3 = sub.add_parser("benchmark", help="Run benchmarks")
    p3.add_argument("--type", choices=["pathfinding", "maze", "rl"], default="pathfinding")
    p3.add_argument("--seed", type=int, default=42)

    # demo
    sub.add_parser("demo", help="Run full demo")

    args = parser.parse_args()

    if args.command == "pathfinding":
        cmd_pathfinding(args)
    elif args.command == "rl":
        cmd_rl(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
