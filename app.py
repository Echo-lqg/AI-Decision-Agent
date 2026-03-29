"""
Application interactive Streamlit — AI Decision Agent.

Fournit une interface web interactive pour :
1. Configurer et visualiser les environnements GridWorld
2. Exécuter et comparer les algorithmes de recherche classiques
3. Entraîner et évaluer les agents RL (Q-Learning, SARSA)
4. Consulter les benchmarks et analyses statistiques
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
from src.benchmark import (
    benchmark_pathfinding, benchmark_mazes, benchmark_rl,
    benchmark_cross_comparison, statistical_tests, summary_table,
)

st.set_page_config(
    page_title="AI Decision Agent — Planification & RL",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Agent Intelligent : Planification de Chemins & Prise de Décision")
st.markdown("""
> **Projet personnel** — Comparaison d'algorithmes de recherche classiques et d'agents
> par apprentissage par renforcement dans un environnement GridWorld configurable.
""")

# ── Sidebar Configuration ──────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

tab = st.sidebar.radio(
    "Mode",
    ["🗺️ Pathfinding", "🤖 Reinforcement Learning", "📊 Benchmark"],
)

st.sidebar.subheader("Environnement")
env_type = st.sidebar.selectbox("Type", ["Grille aléatoire", "Labyrinthe DFS", "Labyrinthe Prim"])
grid_size = st.sidebar.slider("Taille de la grille", 7, 51, 15, step=2)
seed = st.sidebar.number_input("Graine aléatoire", 0, 9999, 42)

if env_type == "Grille aléatoire":
    obstacle_ratio = st.sidebar.slider("Ratio d'obstacles", 0.0, 0.45, 0.2, 0.05)
    swamp_ratio = st.sidebar.slider("Ratio de marécages", 0.0, 0.3, 0.05, 0.05)
    reward_count = st.sidebar.slider("Cellules récompense", 0, 10, 3)
else:
    obstacle_ratio = 0.0
    swamp_ratio = 0.0
    reward_count = 0


@st.cache_data
def create_env(env_type, grid_size, obstacle_ratio, swamp_ratio, reward_count, seed):
    if env_type == "Labyrinthe DFS":
        return generate_maze_dfs(grid_size, grid_size, seed=seed)
    elif env_type == "Labyrinthe Prim":
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
    st.header("Algorithmes de recherche classiques")

    algo_choices = st.multiselect(
        "Sélectionner les algorithmes à comparer",
        list(ALL_ALGORITHMS.keys()),
        default=list(ALL_ALGORITHMS.keys()),
    )

    if st.button("🚀 Lancer les algorithmes", type="primary"):
        results = [ALL_ALGORITHMS[name](env) for name in algo_choices]

        st.subheader("Comparaison côte à côte")
        fig = plot_comparison(env, results)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Cartes de chaleur d'exploration")
        fig = plot_exploration_heatmap(env, results)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Métriques de performance")
        rows = []
        for r in results:
            rows.append({
                "Algorithme": r.algorithm,
                "Chemin trouvé": "✅" if r.found else "❌",
                "Longueur": r.path_length,
                "Coût": f"{r.path_cost:.1f}",
                "Nœuds explorés": r.nodes_explored,
                "Temps (ms)": f"{r.time_seconds * 1000:.3f}",
            })
        st.table(pd.DataFrame(rows))

        st.subheader("Analyse")
        if all(r.found for r in results):
            optimal = min(results, key=lambda r: r.path_cost)
            fastest = min(results, key=lambda r: r.time_seconds)
            most_efficient = min(results, key=lambda r: r.nodes_explored)

            col1, col2, col3 = st.columns(3)
            col1.metric("🏆 Chemin optimal", optimal.algorithm, f"Coût : {optimal.path_cost:.1f}")
            col2.metric("⚡ Plus rapide", fastest.algorithm, f"{fastest.time_seconds*1000:.3f}ms")
            col3.metric("🎯 Plus efficace", most_efficient.algorithm, f"{most_efficient.nodes_explored} nœuds")

            st.markdown(f"""
            **Observations clés :**
            - **A*** utilise la guidance heuristique pour explorer moins de nœuds ({[r for r in results if r.algorithm == 'A*'][0].nodes_explored if any(r.algorithm == 'A*' for r in results) else 'N/A'}) tout en trouvant le chemin optimal.
            - **BFS** garantit le plus court chemin dans les graphes non pondérés mais explore davantage de nœuds.
            - **DFS** est rapide mais trouve souvent des chemins sous-optimaux.
            - **Dijkstra** gère de manière optimale les arêtes pondérées (terrain marécageux).
            """)
        else:
            st.warning("Certains algorithmes n'ont pas trouvé de chemin. Le départ et l'objectif ne sont peut-être pas connectés.")

    else:
        st.info("Configurez l'environnement dans la barre latérale, puis cliquez sur **Lancer les algorithmes**.")
        fig = plot_grid(env, f"GridWorld {grid_size}×{grid_size}")
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# TAB 2: REINFORCEMENT LEARNING
# ══════════════════════════════════════════════════════════════════
elif tab == "🤖 Reinforcement Learning":
    st.header("Agents d'apprentissage par renforcement")

    st.sidebar.subheader("Hyperparamètres RL")
    episodes = st.sidebar.slider("Épisodes d'entraînement", 100, 5000, 1000, 100)
    alpha = st.sidebar.slider("Taux d'apprentissage (α)", 0.01, 0.5, 0.1, 0.01)
    gamma = st.sidebar.slider("Facteur d'actualisation (γ)", 0.5, 1.0, 0.99, 0.01)
    epsilon_decay = st.sidebar.slider("Décroissance d'epsilon", 0.99, 1.0, 0.995, 0.001)

    rl_choices = st.multiselect(
        "Sélectionner les algorithmes RL",
        ["Q-Learning", "SARSA"],
        default=["Q-Learning", "SARSA"],
    )

    if st.button("🧠 Entraîner les agents", type="primary"):
        training_results = []
        progress = st.progress(0)

        for i, algo_name in enumerate(rl_choices):
            with st.spinner(f"Entraînement de {algo_name}..."):
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

        st.subheader("Courbes d'entraînement")
        fig = plot_training_curves(training_results)
        st.pyplot(fig)
        plt.close(fig)

        for result in training_results:
            st.subheader(f"{result.algorithm} — Politique apprise")
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
            c1.metric("Objectif atteint", "✅ Oui" if reached else "❌ Non")
            c2.metric("Longueur du chemin", len(result.policy_path))
            c3.metric("Convergé à l'épisode", result.converged_at or "N/A")

        if len(training_results) == 2:
            st.subheader("Comparaison : Q-Learning vs SARSA")
            st.markdown("""
            | Aspect | Q-Learning | SARSA |
            |--------|-----------|-------|
            | **Règle de mise à jour** | Off-policy (utilise max Q) | On-policy (utilise l'action réellement suivie) |
            | **Exploration** | Plus agressive | Plus conservatrice |
            | **Optimalité** | Converge vers Q* optimal | Converge vers Q dépendant de la politique |
            | **Sûreté** | Peut apprendre des raccourcis risqués | Politiques plus sûres près des pénalités |
            """)

    else:
        st.info("Configurez les hyperparamètres dans la barre latérale, puis cliquez sur **Entraîner les agents**.")
        fig = plot_grid(env, f"GridWorld {grid_size}×{grid_size}")
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# TAB 3: BENCHMARK
# ══════════════════════════════════════════════════════════════════
elif tab == "📊 Benchmark":
    st.header("Benchmarks systématiques & Analyse")

    bench_type = st.selectbox("Type de benchmark", [
        "Recherche : Passage à l'échelle",
        "Recherche : Comparaison de labyrinthes",
        "RL : Q-Learning vs SARSA",
        "Comparaison croisée : Recherche vs RL (mêmes grilles)",
    ])

    if st.button("📈 Lancer le benchmark", type="primary"):
        with st.spinner("Exécution du benchmark... cela peut prendre un moment."):
            if bench_type == "Recherche : Passage à l'échelle":
                df = benchmark_pathfinding(
                    sizes=[11, 15, 21, 31],
                    obstacle_ratios=[0.1, 0.2, 0.3],
                    n_trials=3,
                    seed=seed,
                )

                st.subheader("Résultats bruts")
                st.dataframe(df, use_container_width=True)

                st.subheader("Résumé par algorithme et taille de grille")
                summary = summary_table(df, ["algorithm", "grid_size"])
                st.dataframe(summary, use_container_width=True)

                st.subheader("Nœuds explorés vs Taille de grille")
                fig, ax = plt.subplots(figsize=(10, 5))
                for algo in df["algorithm"].unique():
                    sub = df[df["algorithm"] == algo]
                    means = sub.groupby("grid_size")["nodes_explored"].mean()
                    ax.plot(means.index, means.values, "o-", label=algo, linewidth=2)
                ax.set_xlabel("Taille de la grille")
                ax.set_ylabel("Nœuds explorés (moyenne)")
                ax.set_title("Passage à l'échelle : Nœuds explorés vs Taille")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Temps d'exécution vs Taille de grille")
                fig, ax = plt.subplots(figsize=(10, 5))
                for algo in df["algorithm"].unique():
                    sub = df[df["algorithm"] == algo]
                    means = sub.groupby("grid_size")["time_ms"].mean()
                    ax.plot(means.index, means.values, "o-", label=algo, linewidth=2)
                ax.set_xlabel("Taille de la grille")
                ax.set_ylabel("Temps moyen (ms)")
                ax.set_title("Passage à l'échelle : Temps d'exécution vs Taille")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

            elif bench_type == "Recherche : Comparaison de labyrinthes":
                df = benchmark_mazes(sizes=[11, 15, 21, 31], n_trials=3, seed=seed)

                st.subheader("Résultats")
                st.dataframe(df, use_container_width=True)

                st.subheader("Résumé")
                summary = summary_table(df, ["algorithm", "maze_type"])
                st.dataframe(summary, use_container_width=True)

                st.subheader("Nœuds explorés par type de labyrinthe")
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                for ax, maze_type in zip(axes, df["maze_type"].unique()):
                    sub = df[df["maze_type"] == maze_type]
                    for algo in sub["algorithm"].unique():
                        asub = sub[sub["algorithm"] == algo]
                        means = asub.groupby("grid_size")["nodes_explored"].mean()
                        ax.plot(means.index, means.values, "o-", label=algo, linewidth=2)
                    ax.set_title(maze_type)
                    ax.set_xlabel("Taille de la grille")
                    ax.set_ylabel("Nœuds explorés")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

            elif bench_type == "RL : Q-Learning vs SARSA":
                df = benchmark_rl(
                    grid_size=min(grid_size, 15),
                    obstacle_ratio=0.15,
                    episodes=500,
                    n_trials=3,
                    seed=seed,
                )

                st.subheader("Résultats du benchmark RL")
                st.dataframe(df, use_container_width=True)

            else:
                df = benchmark_cross_comparison(
                    grid_size=min(grid_size, 15),
                    obstacle_ratio=0.15,
                    episodes=1000,
                    n_trials=10,
                    seed=seed,
                )

                st.subheader("Comparaison croisée : Recherche vs RL sur grilles identiques")
                st.dataframe(df, use_container_width=True)

                st.subheader("Coût du chemin par algorithme")
                found_df = df[df["found"] == True]
                if not found_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    algo_order = found_df.groupby("algorithm")["path_cost"].mean().sort_values().index
                    positions = range(len(algo_order))
                    for i, algo in enumerate(algo_order):
                        values = found_df[found_df["algorithm"] == algo]["path_cost"]
                        ax.boxplot(values, positions=[i], widths=0.6)
                    ax.set_xticks(list(positions))
                    ax.set_xticklabels(algo_order, fontsize=10)
                    ax.set_ylabel("Coût du chemin")
                    ax.set_title("Distribution du coût (exécutions réussies uniquement)")
                    ax.grid(True, alpha=0.3, axis="y")
                    st.pyplot(fig)
                    plt.close(fig)

                st.subheader("Tests de significativité statistique")
                st.caption(
                    "Test de Wilcoxon signé (apparié par condition d'environnement, "
                    "non-paramétrique, α = 0.05). "
                    "La comparaison recherche vs RL est intrinsèquement asymétrique : "
                    "la recherche dispose du modèle complet, le RL opère sans modèle."
                )
                st_df = statistical_tests(df)
                if st_df.empty:
                    st.warning("Données insuffisantes. Essayez d'augmenter le nombre d'essais.")
                else:
                    for _, row in st_df.iterrows():
                        p = row["p_value"]
                        r_eff = row["effect_size_r"]

                        if p < 0.01:
                            level, label = "high", "Très significatif"
                        elif p < 0.05:
                            level, label = "medium", "Significatif"
                        else:
                            level, label = "low", "Non significatif"

                        if r_eff >= 0.5:
                            eff_label = "fort"
                        elif r_eff >= 0.3:
                            eff_label = "moyen"
                        else:
                            eff_label = "faible"

                        _LEVEL_ICON = {"high": "\U0001f7e2", "medium": "\U0001f7e1", "low": "\U0001f534"}
                        _LEVEL_THRESHOLD = {"high": "p < 0.01", "medium": "p < 0.05", "low": "p \u2265 0.05"}
                        icon = _LEVEL_ICON[level]
                        threshold = _LEVEL_THRESHOLD[level]

                        with st.container():
                            st.markdown(f"**{row['hypothesis']}**")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric(row["group_a"], f"{row['mean_a']:.2f}")
                            col2.metric(row["group_b"], f"{row['mean_b']:.2f}")
                            col3.metric("p-value", f"{p:.4f}")
                            col4.metric("Effet (r)", f"{r_eff:.3f}", delta=eff_label)
                            st.markdown(
                                f"> {icon} {threshold} — {label} &nbsp;|&nbsp; "
                                f"\u0394 = {row['mean_diff']:.2f} &nbsp;|&nbsp; "
                                f"W = {row['W_statistic']:.0f} &nbsp;|&nbsp; "
                                f"n = {row['n_pairs']} paires"
                            )
                            st.divider()

    else:
        st.info("Sélectionnez un type de benchmark et cliquez sur **Lancer le benchmark**.")

# ── Pied de page ────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("""
**AI Decision Agent**
*Planification de chemins & Apprentissage par renforcement*

Projet personnel d'autoformation
""")
