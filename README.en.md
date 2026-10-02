[Français](README.md) | English

# AI Decision Agent — Path Planning & Reinforcement Learning

> **Personal self-study project** — Comparative study of classical search
> algorithms (BFS, DFS, Dijkstra, A\*) and reinforcement learning agents
> (Q-Learning, SARSA) for autonomous navigation in a configurable GridWorld
> environment with weighted terrain.

---

## Context & Objectives

This project grew out of a personal curiosity about **autonomous decision-making**
problems — a topic I had not covered in my first year of the Master's program (M1).
Starting from the reference textbooks (Russell & Norvig, Sutton & Barto), I set out
to understand and implement on my own two fundamental approaches:

- **Graph search**: how can an agent find an optimal path when it has complete
  knowledge of its environment?
- **Reinforcement learning**: how can an agent learn to navigate *without* a prior
  model, purely through trial and error?

Beyond the implementation, the goal was to **experimentally and rigorously compare**
these two paradigms — with explicit hypotheses, identical environments for each
comparison, and statistical significance tests — in order to better understand their
strengths, their limitations, and the situations in which one is preferable to the
other.

---

## Project Description

### 1. Classical Search Algorithms

Four algorithms implemented from scratch (without a graph library):

| Algorithm | Type | Optimality | Complexity |
|-----------|------|-----------|------------|
| **BFS** | Uninformed | Optimal (uniform cost) | O(V + E) |
| **DFS** | Uninformed | Not optimal | O(V + E) |
| **Dijkstra** | Cost-informed | Optimal (costs ≥ 0) | O((V+E) log V) |
| **A\*** | Heuristic (Manhattan) | Optimal (admissible h) | O((V+E) log V) |

### 2. Tabular Reinforcement Learning

| Agent | Update | Property |
|-------|--------|----------|
| **Q-Learning** | Off-policy: `max_a Q(s',a)` | Can converge to optimal Q\* under standard conditions; off-policy exploration |
| **SARSA** | On-policy: `Q(s',a')` | More cautious policies, penalties better avoided |

### 3. GridWorld Environment

The environment is modeled as an implicit weighted graph:

- Configurable grids (size from 7×7 to 51×51)
- **Weighted terrain**: swamp cells (traversal cost ×3)
- **Reward system**: bonus cells (+10), living penalty (-1), goal (+100)
- **Procedural maze generation**: recursive DFS and Prim's algorithm

---

## Project Architecture

```
AI-Decision-Agent/
├── README.md                 # French documentation
├── README.en.md              # English documentation
├── requirements.txt          # Python dependencies
├── main.py                   # CLI entry point (demo, benchmarks)
├── app.py                    # Interactive web interface (Streamlit)
└── src/
    ├── __init__.py
    ├── environment.py        # GridWorld model, dynamics, maze generation
    ├── pathfinding.py        # BFS, DFS, Dijkstra, A* — implemented from scratch
    ├── rl_agent.py           # Tabular Q-Learning & SARSA, epsilon-greedy
    ├── visualizer.py         # Visualization: grids, heatmaps, curves, value maps
    └── benchmark.py          # Systematic experiments and statistical analysis
```

Design principles:
- Separation between environment, algorithms, and visualization
- Common interface (`SearchResult`, `TrainingResult`) to facilitate comparison
- Modular code, documented with docstrings

---

## Installation & Usage

### Prerequisites
- Python 3.9+

### Installation
```bash
git clone https://github.com/Echo-lqg/AI-Decision-Agent.git
cd AI-Decision-Agent
pip install -r requirements.txt
```

### CLI — Command Line
```bash
# Full demo (pathfinding + RL + visualizations)
python main.py demo

# Comparison of the search algorithms
python main.py pathfinding --size 21 --seed 42

# Pathfinding in a generated maze
python main.py pathfinding --size 21 --maze dfs

# Training the RL agents
python main.py rl --size 11 --episodes 2000

# Systematic benchmarks
python main.py benchmark --type pathfinding
python main.py benchmark --type maze
python main.py benchmark --type rl

# Cross comparison (search vs RL, same grids) + statistical tests
python main.py benchmark --type cross
```

### Interactive Web Interface
```bash
streamlit run app.py
```

The Streamlit interface allows you to:
- Configure the environment in real time (size, obstacle density, maze type)
- Run and visually compare the search algorithms
- Train the RL agents with adjustable hyperparameters (α, γ, ε-decay)
- Run benchmarks and view the results as tables and charts

---

## Experimental Protocol & Hypotheses

### Hypotheses

| # | Hypothesis | Metric | Test |
|---|-----------|--------|------|
| **H1** | A\* explores fewer nodes than BFS (heuristic guidance) | `nodes_explored` | Wilcoxon signed-rank |
| **H2** | Dijkstra finds lower-cost paths than BFS on weighted terrain | `path_cost` | Wilcoxon signed-rank |
| **H3** | Q-Learning converges faster than SARSA | `converged_at` | Wilcoxon signed-rank |
| **H4** | SARSA produces paths with a cost lower than or equal to Q-Learning | `path_cost` | Wilcoxon signed-rank |

### Protocol

- **Identical environments**: each comparison (pathfinding vs pathfinding, RL vs RL,
  and above all pathfinding vs RL) is carried out on **exactly the same generated grid**,
  including obstacles and weighted terrain (swamps, cost = 3).
- **Paired data**: each trial produces a pair of observations (same grid →
  result of algorithm A and result of algorithm B), which justifies a **paired** test.
- **Multiple repetitions**: each configuration is tested over *n* independent trials
  (different seeds) to reduce variance.
- **Acknowledged asymmetry**: the cross comparison (search vs RL) highlights
  performance differences, but remains **intrinsically asymmetric**: classical search
  algorithms have complete knowledge of the environment (*full model*), whereas
  reinforcement learning operates without a model (*model-free*) and must discover the
  structure through exploration. This distinction is fundamental in artificial
  intelligence, and the results must be interpreted with this difference in paradigm
  in mind.

### Statistical Analysis

We use the **Wilcoxon signed-rank test** (non-parametric) to assess the statistical
significance of the observed differences between algorithms (α = 0.05).

This choice is justified by:
1. **Paired design** — observations are paired by environment condition
   (trial × grid_size × obstacle_ratio), which rules out tests for independent
   samples (Mann-Whitney U, independent t-test).
2. **Non-normal distribution** — the metrics (nodes explored, path cost,
   convergence episode) do not necessarily follow a normal distribution,
   which rules out the parametric paired t-test.
3. **Effect size** — in addition to the p-value, we report *r* = |Z| / √N
   to quantify the practical magnitude of the difference, independently of the sample
   size. Z is obtained directly from SciPy's normal approximation
   (`method="approx"`), which includes the correction for ties.

```bash
# Run the cross comparison with statistical tests
python main.py benchmark --type cross
```

---

## Results & Observations

### Classical Search

- **A\*** generally explores fewer nodes than BFS thanks to the Manhattan heuristic, while preserving optimality.
- **Dijkstra** proves indispensable when the terrain is weighted (swamps), where BFS no longer guarantees optimality.
- **DFS** is fast but produces paths that are often much longer than the optimal one.
- The performance gap between A\* and BFS widens as the grid size increases.

### Reinforcement Learning

- **Q-Learning** (off-policy) tends to converge faster to an efficient policy, but may take risky trajectories.
- **SARSA** (on-policy) learns more cautious policies, avoiding high-penalty areas more often.
- Convergence of both agents requires a sufficient number of episodes (~500-1000 depending on grid complexity).

### Summary: Classical Search vs RL

| Criterion | Classical Search | Reinforcement Learning |
|-----------|-----------------|------------------------|
| **Required model** | Complete (explicit graph) | None (trial-and-error learning) |
| **Optimality** | Guaranteed (A\*, Dijkstra) | Asymptotic (convergence under conditions) |
| **Adaptability** | Recomputation if the environment changes | Adaptation through retraining |
| **Cost** | Per query (real time) | Initial training phase |

> Detailed results (p-values, effect sizes) are generated automatically
> by `python main.py benchmark --type cross` and saved in `output_statistical_tests.csv`.

---

## What I Learned from This Project

This project, carried out on my own outside of my M1 curriculum, allowed me to:

- **Gain hands-on understanding** of graph search algorithms by implementing them from scratch, and understand why A\* is so widely used in practice.
- **Approach reinforcement learning** through tabular cases (Q-Learning, SARSA), and grasp the fundamental distinction between on-policy and off-policy approaches.
- **Formalize a problem** by modeling it as an MDP (states, actions, transitions, rewards).
- **Develop an experimental approach**: systematic benchmarks, statistical analysis, quantitative comparison between approaches.
- **Connect what I have learned in psychology to computational models of decision-making**: the notions of reward, exploration/exploitation, and trial-and-error learning find a direct echo in the behavioral theories I have studied, and this project allowed me to formalize them mathematically.
- **Identify my limitations**: this project remains within the tabular framework; I would like to later explore methods with function approximation (Deep RL) and more complex environments.

---

## Technologies

| Technology | Role |
|------------|------|
| **Python 3.9+** | Main language |
| **NumPy** | Numerical computation, Q-tables |
| **Matplotlib** | Visualization (grids, heatmaps, training curves, value maps) |
| **Streamlit** | Interactive web interface |
| **Pandas** | Statistical analysis of benchmarks |
| **SciPy** | Statistical testing (Wilcoxon signed-rank) |

---

## References

- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach* (4th ed., 2020)
- Sutton, R. & Barto, A. *Reinforcement Learning: An Introduction* (2nd ed., 2018)
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*. IEEE Transactions on Systems Science and Cybernetics.
- Watkins, C. J. & Dayan, P. (1992). *Q-Learning*. Machine Learning, 8(3-4), 279-292.
- Rummery, G. A. & Niranjan, M. (1994). *On-Line Q-Learning Using Connectionist Systems*. Technical Report CUED/F-INFENG/TR 166, Cambridge University.

---

## Author

**LIU Qiange**
