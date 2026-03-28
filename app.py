"""
Streamlit Interactive Application — AI Decision Agent.

Provides an interactive web interface to:
1. Configure and visualize GridWorld environments
2. Run and compare classical pathfinding algorithms
3. Train and evaluate RL agents (Q-Learning, SARSA)
4. View benchmarks and analysis
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.environment import GridWorld, generate_maze_dfs, generate_maze_prim
from src.pathfinding import ALL_ALGORITHMS
from src.rl_agent import QLearningAgent, SARSAAgent
from src.visualizer import (
    plot_grid, plot_comparison,
    plot_exploration_heatmap, plot_training_curves,
    plot_value_map, plot_rl_path_on_grid,
)
from src.benchmark import benchmark_pathfinding, benchmark_mazes, benchmark_rl, summary_table

st.set_page_config(
    page_title="AI Decision Agent — Path Planning & RL",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Intelligent Agent: Path Planning & Decision Making")
st.markdown("""
> **Projet AI2D** — Comparaison d'algorithmes de recherche classiques et d'agents
> par apprentissage par renforcement dans un environnement GridWorld configurable.
""")

# ── Sidebar Configuration ──────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

tab = st.sidebar.radio(
    "Mode",
    ["🗺️ Pathfinding", "🤖 Reinforcement Learning", "📊 Benchmark"],
)

st.sidebar.subheader("Environment")
env_type = st.sidebar.selectbox("Type", ["Random Grid", "DFS Maze", "Prim Maze"])
grid_size = st.sidebar.slider("Grid Size", 7, 51, 15, step=2)
seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)

if env_type == "Random Grid":
    obstacle_ratio = st.sidebar.slider("Obstacle Ratio", 0.0, 0.45, 0.2, 0.05)
    swamp_ratio = st.sidebar.slider("Swamp Ratio", 0.0, 0.3, 0.05, 0.05)
    reward_count = st.sidebar.slider("Reward Cells", 0, 10, 3)
else:
    obstacle_ratio = 0.0
    swamp_ratio = 0.0
    reward_count = 0


@st.cache_data
def create_env(env_type, grid_size, obstacle_ratio, swamp_ratio, reward_count, seed):
    if env_type == "DFS Maze":
        return generate_maze_dfs(grid_size, grid_size, seed=seed)
    elif env_type == "Prim Maze":
        return generate_maze_prim(grid_size, grid_size, seed=seed)
    else:
        env = GridWorld(grid_size, grid_size)
        env.add_random_obstacles(ratio=obstacle_ratio, seed=seed)
        env.add_random_swamps(ratio=swamp_ratio, seed=seed + 1)
        env.add_random_rewards(count=reward_count, seed=seed + 2)
        return env


env = create_env(env_type, grid_size, obstacle_ratio, swamp_ratio, reward_count, seed)

# ══════════════════════════════════════════════════════════════════
# TAB 1: PATHFINDING
# ══════════════════════════════════════════════════════════════════
if tab == "🗺️ Pathfinding":
    st.header("Classical Pathfinding Algorithms")

    algo_choices = st.multiselect(
        "Select algorithms to compare",
        list(ALL_ALGORITHMS.keys()),
        default=list(ALL_ALGORITHMS.keys()),
    )

    if st.button("🚀 Run Algorithms", type="primary"):
        results = [ALL_ALGORITHMS[name](env) for name in algo_choices]

        st.subheader("Side-by-Side Comparison")
        fig = plot_comparison(env, results)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Exploration Heatmaps")
        fig = plot_exploration_heatmap(env, results)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Performance Metrics")
        rows = []
        for r in results:
            rows.append({
                "Algorithm": r.algorithm,
                "Found Path": "✅" if r.found else "❌",
                "Path Length": r.path_length,
                "Path Cost": f"{r.path_cost:.1f}",
                "Nodes Explored": r.nodes_explored,
                "Time (ms)": f"{r.time_seconds * 1000:.3f}",
            })
        st.table(pd.DataFrame(rows))

        st.subheader("Analysis")
        if all(r.found for r in results):
            optimal = min(results, key=lambda r: r.path_cost)
            fastest = min(results, key=lambda r: r.time_seconds)
            most_efficient = min(results, key=lambda r: r.nodes_explored)

            col1, col2, col3 = st.columns(3)
            col1.metric("🏆 Optimal Path", optimal.algorithm, f"Cost: {optimal.path_cost:.1f}")
            col2.metric("⚡ Fastest", fastest.algorithm, f"{fastest.time_seconds*1000:.3f}ms")
            col3.metric("🎯 Most Efficient", most_efficient.algorithm, f"{most_efficient.nodes_explored} nodes")

            st.markdown(f"""
            **Key Observations:**
            - **A*** uses heuristic guidance to explore fewer nodes ({[r for r in results if r.algorithm == 'A*'][0].nodes_explored if any(r.algorithm == 'A*' for r in results) else 'N/A'}) while finding the optimal path.
            - **BFS** guarantees shortest path in unweighted graphs but explores more nodes.
            - **DFS** is fast but often finds suboptimal paths.
            - **Dijkstra** handles weighted edges (swamp terrain) optimally.
            """)
        else:
            st.warning("Some algorithms did not find a path. The start and goal may not be connected.")

    else:
        st.info("Configure the environment in the sidebar, then click **Run Algorithms**.")
        fig = plot_grid(env, f"GridWorld {grid_size}×{grid_size}")
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# TAB 2: REINFORCEMENT LEARNING
# ══════════════════════════════════════════════════════════════════
elif tab == "🤖 Reinforcement Learning":
    st.header("Reinforcement Learning Agents")

    st.sidebar.subheader("RL Hyperparameters")
    episodes = st.sidebar.slider("Training Episodes", 100, 5000, 1000, 100)
    alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 0.5, 0.1, 0.01)
    gamma = st.sidebar.slider("Discount Factor (γ)", 0.5, 1.0, 0.99, 0.01)
    epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.99, 1.0, 0.995, 0.001)

    rl_choices = st.multiselect(
        "Select RL algorithms",
        ["Q-Learning", "SARSA"],
        default=["Q-Learning", "SARSA"],
    )

    if st.button("🧠 Train Agents", type="primary"):
        training_results = []
        progress = st.progress(0)

        for i, algo_name in enumerate(rl_choices):
            with st.spinner(f"Training {algo_name}..."):
                if algo_name == "Q-Learning":
                    agent = QLearningAgent(
                        env.copy(), alpha=alpha, gamma=gamma,
                        epsilon_decay=epsilon_decay, seed=seed,
                    )
                else:
                    agent = SARSAAgent(
                        env.copy(), alpha=alpha, gamma=gamma,
                        epsilon_decay=epsilon_decay, seed=seed,
                    )
                result = agent.train(episodes=episodes)
                training_results.append(result)
            progress.progress((i + 1) / len(rl_choices))

        st.subheader("Training Curves")
        fig = plot_training_curves(training_results)
        st.pyplot(fig)
        plt.close(fig)

        for result in training_results:
            st.subheader(f"{result.algorithm} — Learned Policy")
            col1, col2 = st.columns(2)

            with col1:
                fig = plot_rl_path_on_grid(env, result)
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                fig = plot_value_map(env, result.q_table, f"{result.algorithm} Value Map")
                st.pyplot(fig)
                plt.close(fig)

            reached = result.policy_path[-1] == env.goal if result.policy_path else False
            c1, c2, c3 = st.columns(3)
            c1.metric("Reached Goal", "✅ Yes" if reached else "❌ No")
            c2.metric("Policy Path Length", len(result.policy_path))
            c3.metric("Converged at Episode", result.converged_at or "N/A")

        if len(training_results) == 2:
            st.subheader("Comparison: Q-Learning vs SARSA")
            st.markdown("""
            | Aspect | Q-Learning | SARSA |
            |--------|-----------|-------|
            | **Update Rule** | Off-policy (uses max Q) | On-policy (uses actual next action) |
            | **Exploration** | More aggressive | More conservative |
            | **Optimality** | Converges to optimal Q* | Converges to policy-dependent Q |
            | **Safety** | May learn risky shortcuts | Safer policies near penalties |
            """)

    else:
        st.info("Configure hyperparameters in the sidebar, then click **Train Agents**.")
        fig = plot_grid(env, f"GridWorld {grid_size}×{grid_size}")
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# TAB 3: BENCHMARK
# ══════════════════════════════════════════════════════════════════
elif tab == "📊 Benchmark":
    st.header("Systematic Benchmark & Analysis")

    bench_type = st.selectbox("Benchmark Type", [
        "Pathfinding: Grid Size Scaling",
        "Pathfinding: Maze Comparison",
        "RL: Q-Learning vs SARSA",
    ])

    if st.button("📈 Run Benchmark", type="primary"):
        with st.spinner("Running benchmark... this may take a moment."):
            if bench_type == "Pathfinding: Grid Size Scaling":
                df = benchmark_pathfinding(
                    sizes=[11, 15, 21, 31],
                    obstacle_ratios=[0.1, 0.2, 0.3],
                    n_trials=3,
                    seed=seed,
                )

                st.subheader("Raw Results")
                st.dataframe(df, use_container_width=True)

                st.subheader("Summary by Algorithm & Grid Size")
                summary = summary_table(df, ["algorithm", "grid_size"])
                st.dataframe(summary, use_container_width=True)

                st.subheader("Nodes Explored vs Grid Size")
                fig, ax = plt.subplots(figsize=(10, 5))
                for algo in df["algorithm"].unique():
                    sub = df[df["algorithm"] == algo]
                    means = sub.groupby("grid_size")["nodes_explored"].mean()
                    ax.plot(means.index, means.values, "o-", label=algo, linewidth=2)
                ax.set_xlabel("Grid Size")
                ax.set_ylabel("Avg Nodes Explored")
                ax.set_title("Scalability: Nodes Explored vs Grid Size")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Execution Time vs Grid Size")
                fig, ax = plt.subplots(figsize=(10, 5))
                for algo in df["algorithm"].unique():
                    sub = df[df["algorithm"] == algo]
                    means = sub.groupby("grid_size")["time_ms"].mean()
                    ax.plot(means.index, means.values, "o-", label=algo, linewidth=2)
                ax.set_xlabel("Grid Size")
                ax.set_ylabel("Avg Time (ms)")
                ax.set_title("Scalability: Execution Time vs Grid Size")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

            elif bench_type == "Pathfinding: Maze Comparison":
                df = benchmark_mazes(sizes=[11, 15, 21, 31], n_trials=3, seed=seed)

                st.subheader("Results")
                st.dataframe(df, use_container_width=True)

                st.subheader("Summary")
                summary = summary_table(df, ["algorithm", "maze_type"])
                st.dataframe(summary, use_container_width=True)

                st.subheader("Nodes Explored by Maze Type")
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                for ax, maze_type in zip(axes, df["maze_type"].unique()):
                    sub = df[df["maze_type"] == maze_type]
                    for algo in sub["algorithm"].unique():
                        asub = sub[sub["algorithm"] == algo]
                        means = asub.groupby("grid_size")["nodes_explored"].mean()
                        ax.plot(means.index, means.values, "o-", label=algo, linewidth=2)
                    ax.set_title(maze_type)
                    ax.set_xlabel("Grid Size")
                    ax.set_ylabel("Nodes Explored")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

            else:
                df = benchmark_rl(
                    grid_size=min(grid_size, 15),
                    obstacle_ratio=0.15,
                    episodes=500,
                    n_trials=3,
                    seed=seed,
                )

                st.subheader("RL Benchmark Results")
                st.dataframe(df, use_container_width=True)

    else:
        st.info("Select a benchmark type and click **Run Benchmark**.")

# ── Footer ──────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("""
**AI Decision Agent**
*Path Planning & Reinforcement Learning*

Projet pour Sorbonne AI2D
""")
