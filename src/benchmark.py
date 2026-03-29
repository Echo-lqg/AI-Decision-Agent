"""
Module de benchmarks et d'analyse statistique.

Exécute des expériences systématiques comparant les algorithmes de recherche
et les agents RL sur différentes tailles de grille, densités d'obstacles et
types de labyrinthes. Toutes les comparaisons utilisent des environnements
identiques (mêmes grilles, mêmes obstacles, même terrain pondéré) pour
garantir l'équité.

Produit des DataFrames, des statistiques descriptives et des tests de
significativité statistique.

Hypothèses testées :
  H1 : A* explore moins de nœuds que BFS (guidance heuristique).
  H2 : Dijkstra trouve des chemins de moindre coût que BFS (terrain pondéré).
  H3 : Q-Learning converge plus vite (moins d'épisodes) que SARSA.
  H4 : SARSA produit des chemins de coût ≤ Q-Learning (politique conservatrice).

Méthodologie statistique :
  - Plan apparié : les observations sont appariées par condition d'environnement
    (trial × grid_size × obstacle_ratio × maze_type) car chaque algorithme est
    évalué sur exactement la même grille.
  - Test de Wilcoxon signé (signed-rank) : test non-paramétrique adapté aux
    données appariées sans hypothèse de normalité. Seuil α = 0.05.
  - Taille d'effet : r = |Z| / √N, où Z est le z-score de l'approximation
    normale (tie-corrected, via scipy method="approx").
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .environment import GridWorld, generate_maze_dfs, generate_maze_prim
from .pathfinding import ALL_ALGORITHMS, SearchResult, run_all
from .rl_agent import QLearningAgent, SARSAAgent, TrainingResult


# ── Benchmark recherche de chemin ────────────────────────────────

def benchmark_pathfinding(
    sizes: List[int] = None,
    obstacle_ratios: List[float] = None,
    n_trials: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Exécute les algorithmes de recherche sur différentes tailles de grille
    et densités d'obstacles. Chaque algorithme est testé sur exactement
    la même grille générée par essai.
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


# ── Benchmark labyrinthes ───────────────────────────────────────

def benchmark_mazes(
    sizes: List[int] = None,
    n_trials: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Recherche de chemin sur des labyrinthes générés (DFS et Prim)."""
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


# ── Benchmark RL ────────────────────────────────────────────────

def benchmark_rl(
    grid_size: int = 11,
    obstacle_ratio: float = 0.15,
    episodes: int = 1000,
    n_trials: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare Q-Learning et SARSA sur le même environnement par essai."""
    records = []

    for trial in range(n_trials):
        trial_seed = seed + trial
        env = GridWorld(grid_size, grid_size)
        env.add_random_obstacles(ratio=obstacle_ratio, seed=trial_seed)
        env.add_random_swamps(ratio=0.05, seed=trial_seed + 1)

        for AgentClass, name in [(QLearningAgent, "Q-Learning"), (SARSAAgent, "SARSA")]:
            agent = AgentClass(env.copy(), seed=trial_seed)
            t0 = time.perf_counter()
            result = agent.train(episodes=episodes)
            elapsed = time.perf_counter() - t0

            final_reward = (
                np.mean(result.episode_rewards[-100:])
                if len(result.episode_rewards) >= 100
                else np.mean(result.episode_rewards)
            )
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


# ── Comparaison croisée : recherche vs RL sur grilles identiques ─

def benchmark_cross_comparison(
    grid_size: int = 11,
    obstacle_ratio: float = 0.15,
    episodes: int = 1000,
    n_trials: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Exécute les algorithmes de recherche ET les agents RL sur exactement
    les mêmes environnements. Cela permet une comparaison équitable du
    coût et de la longueur du chemin entre les deux paradigmes.

    Note : cette comparaison reste intrinsèquement asymétrique — la
    recherche classique dispose d'une connaissance complète de la grille
    (full model), alors que le RL opère sans modèle (model-free) et doit
    découvrir la structure par exploration.
    """
    records = []

    for trial in range(n_trials):
        trial_seed = seed + trial
        env = GridWorld(grid_size, grid_size)
        env.add_random_obstacles(ratio=obstacle_ratio, seed=trial_seed)
        env.add_random_swamps(ratio=0.05, seed=trial_seed + 1)

        search_results = run_all(env)
        for sr in search_results:
            records.append({
                "trial": trial,
                "family": "search",
                "algorithm": sr.algorithm,
                "found": sr.found,
                "path_length": sr.path_length,
                "path_cost": sr.path_cost,
                "nodes_explored": sr.nodes_explored,
            })

        for AgentClass, name in [(QLearningAgent, "Q-Learning"), (SARSAAgent, "SARSA")]:
            agent = AgentClass(env.copy(), seed=trial_seed)
            result = agent.train(episodes=episodes)
            reached = result.policy_path[-1] == env.goal if result.policy_path else False

            rl_cost = 0.0
            if result.policy_path and reached:
                rl_cost = sum(
                    env.step_cost(r, c) for r, c in result.policy_path[1:]
                )

            records.append({
                "trial": trial,
                "family": "rl",
                "algorithm": name,
                "found": reached,
                "path_length": len(result.policy_path) if reached else 0,
                "path_cost": rl_cost if reached else 0.0,
                "nodes_explored": np.nan,
            })

    return pd.DataFrame(records)


# ── Tests de significativité statistique ─────────────────────────

def statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exécute les tests d'hypothèses sur les résultats de benchmark.

    Utilise le test de Wilcoxon signé (paired, non-paramétrique) car
    chaque trial utilise la même grille pour tous les algorithmes :
    les observations sont appariées par condition d'environnement.

    Le merge s'effectue sur TOUTES les colonnes de condition
    (trial, grid_size, obstacle_ratio, maze_type…) afin d'éviter
    les faux appariements quand le DataFrame contient plusieurs
    configurations de grille.

    Retourne un DataFrame avec les résultats incluant les p-values
    et les tailles d'effet (r = Z / sqrt(N)).
    """
    # Colonnes définissant un environnement identique pour l'appariement
    _CONDITION_COLS = {"trial", "grid_size", "obstacle_ratio", "maze_type"}

    test_results = []

    def _paired_test(name_a: str, name_b: str, metric: str,
                     hypothesis: str, subset: pd.DataFrame = None):
        """Wilcoxon signed-rank sur les paires (condition-matched)."""
        # Permet de restreindre l'analyse aux comparaisons valides
        # (ex. : uniquement les exécutions ayant atteint le goal)
        source = subset if subset is not None else df

        pair_keys = [c for c in source.columns if c in _CONDITION_COLS]
        if not pair_keys:
            return

        keep_cols = pair_keys + [metric]
        grp_a = source[source["algorithm"] == name_a][keep_cols].dropna()
        grp_b = source[source["algorithm"] == name_b][keep_cols].dropna()

        paired = grp_a.merge(grp_b, on=pair_keys, suffixes=("_a", "_b"))
        if len(paired) < 5:
            return

        diff = paired[f"{metric}_a"] - paired[f"{metric}_b"]
        # Wilcoxon est indéfini quand toutes les différences sont nulles :
        # on retourne un résultat neutre plutôt que de lever une exception
        if (diff == 0).all():
            test_results.append({
                "hypothesis": hypothesis,
                "test": "Wilcoxon signed-rank",
                "metric": metric,
                "group_a": name_a,
                "group_b": name_b,
                "n_pairs": len(paired),
                "mean_a": paired[f"{metric}_a"].mean(),
                "mean_b": paired[f"{metric}_b"].mean(),
                "mean_diff": 0.0,
                "W_statistic": float("nan"),
                "p_value": 1.0,
                "effect_size_r": 0.0,
                "significant_005": False,
                "significant_01": False,
            })
            return

        res = stats.wilcoxon(
            paired[f"{metric}_a"], paired[f"{metric}_b"],
            alternative="less",
            method="approx",
        )
        n = len(paired)

        z = getattr(res, "zstatistic", None)
        if z is not None:
            r = abs(z) / np.sqrt(n)
            r_note = "z from scipy approx (tie-corrected)"
        else:
            r = float("nan")
            r_note = "zstatistic not available in this scipy version"

        test_results.append({
            "hypothesis": hypothesis,
            "test": "Wilcoxon signed-rank (approx)",
            "metric": metric,
            "group_a": name_a,
            "group_b": name_b,
            "n_pairs": n,
            "mean_a": paired[f"{metric}_a"].mean(),
            "mean_b": paired[f"{metric}_b"].mean(),
            "mean_diff": diff.mean(),
            "W_statistic": res.statistic,
            "p_value": res.pvalue,
            "effect_size_r": round(r, 3) if not np.isnan(r) else float("nan"),
            "effect_size_note": r_note,
            "significant_005": res.pvalue < 0.05,
            "significant_01": res.pvalue < 0.01,
        })

    # H1 : A* explore moins de nœuds que BFS
    if "A*" in df["algorithm"].values and "BFS" in df["algorithm"].values:
        _paired_test("A*", "BFS", "nodes_explored",
                     "H1 : A* explore moins de nœuds que BFS")

    # H2 : Dijkstra trouve des chemins de moindre coût que BFS
    if "Dijkstra" in df["algorithm"].values and "BFS" in df["algorithm"].values:
        found_df = df[df["found"] == True]
        if not found_df.empty:
            _paired_test("Dijkstra", "BFS", "path_cost",
                         "H2 : Dijkstra trouve des chemins de moindre coût que BFS (terrain pondéré)",
                         subset=found_df)

    # H3 : Q-Learning converge plus vite que SARSA
    if "Q-Learning" in df["algorithm"].values and "SARSA" in df["algorithm"].values:
        if "converged_at" in df.columns:
            converged = df[df["converged_at"].notna()]
            if not converged.empty:
                _paired_test("Q-Learning", "SARSA", "converged_at",
                             "H3 : Q-Learning converge plus tôt que SARSA",
                             subset=converged)

        # H4 : SARSA produit des chemins de coût ≤ Q-Learning
        if "path_cost" in df.columns:
            rl_src = df[(df["family"] == "rl") & (df["found"] == True)] if "family" in df.columns else df[df["found"] == True]
            if not rl_src.empty:
                _paired_test("SARSA", "Q-Learning", "path_cost",
                             "H4 : SARSA produit des chemins de coût ≤ Q-Learning",
                             subset=rl_src)

    return pd.DataFrame(test_results)


# ── Tableau récapitulatif ────────────────────────────────────────

def summary_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Calcule les statistiques descriptives groupées par colonnes."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg_cols = [c for c in numeric_cols if c not in group_cols and c != "trial"]
    summary = df.groupby(group_cols)[agg_cols].agg(["mean", "median", "std", "min", "max"]).round(3)

    if "found" in df.columns:
        success = df.groupby(group_cols)["found"].mean().round(3)
        success.name = ("found", "success_rate")
        summary = summary.join(pd.DataFrame(success))

    return summary
