# AI Decision Agent — Path Planning & Reinforcement Learning


> Comparaison systématique d'algorithmes de recherche classiques et d'agents
> par apprentissage par renforcement pour la navigation autonome dans un
> environnement GridWorld configurable.

---

## Description du Projet

Ce projet implémente et compare **deux familles d'approches** pour la résolution
du problème de navigation dans un environnement à grille :

### 1. Algorithmes de Recherche Classiques
| Algorithme | Type | Optimalité | Complexité |
|-----------|------|-----------|------------|
| **BFS** | Non-informé | Optimal (poids uniforme) | O(V + E) |
| **DFS** | Non-informé | Non optimal | O(V + E) |
| **Dijkstra** | Informé par coût | Optimal (poids quelconques) | O((V+E) log V) |
| **A*** | Informé (heuristique) | Optimal (heuristique admissible) | O(E log V) |

### 2. Apprentissage par Renforcement (RL)
| Agent | Politique | Caractéristique |
|-------|----------|----------------|
| **Q-Learning** | Off-policy | Converge vers Q* optimal, exploration agressive |
| **SARSA** | On-policy | Plus conservateur, politiques plus sûres |

### Environnement GridWorld
- Grilles configurables (taille, densité d'obstacles)
- **Terrain pondéré** : cases marécage (coût ×3)
- **Cases récompense** : bonus positifs
- **Génération de labyrinthes** : DFS récursif, algorithme de Prim

---

## Architecture du Projet

```
AI-Decision-Agent/
├── README.md                 # Ce fichier
├── requirements.txt          # Dépendances Python
├── main.py                   # Point d'entrée CLI
├── app.py                    # Application web Streamlit
└── src/
    ├── __init__.py
    ├── environment.py        # GridWorld + génération de labyrinthes
    ├── pathfinding.py        # BFS, DFS, Dijkstra, A*
    ├── rl_agent.py           # Q-Learning, SARSA
    ├── visualizer.py         # Visualisation matplotlib
    └── benchmark.py          # Benchmarks systématiques
```

---

## Installation & Utilisation

### Prérequis
- Python 3.9+

### Installation
```bash
cd AI-Decision-Agent
pip install -r requirements.txt
```

### CLI — Ligne de Commande
```bash
# Demo complète
python main.py demo

# Comparaison des algorithmes de recherche
python main.py pathfinding --size 21 --seed 42

# Pathfinding dans un labyrinthe
python main.py pathfinding --size 21 --maze dfs

# Entraînement RL
python main.py rl --size 11 --episodes 2000

# Benchmarks
python main.py benchmark --type pathfinding
python main.py benchmark --type maze
python main.py benchmark --type rl
```

### Application Web Interactive
```bash
streamlit run app.py
```
L'interface permet de :
- Configurer l'environnement en temps réel (taille, obstacles, type de labyrinthe)
- Lancer et comparer visuellement les algorithmes
- Entraîner les agents RL avec des hyperparamètres ajustables
- Exécuter des benchmarks systématiques

---

## Résultats Clés & Analyse

### Recherche Classique
- **A*** explore significativement moins de nœuds que BFS/DFS grâce à l'heuristique Manhattan, tout en garantissant l'optimalité.
- **Dijkstra** est essentiel lorsque le terrain est pondéré (marécages), où BFS ne donne plus de résultat optimal.
- **DFS** est rapide mais produit des chemins sous-optimaux, parfois très longs.
- La **scalabilité** de A* est supérieure : l'écart de performance s'accentue avec la taille de la grille.

### Apprentissage par Renforcement
- **Q-Learning** (off-policy) converge plus rapidement vers la politique optimale mais peut apprendre des raccourcis risqués.
- **SARSA** (on-policy) produit des politiques plus prudentes, évitant les zones à risque.
- Les deux agents nécessitent un nombre suffisant d'épisodes pour converger sur des grilles complexes.

### Comparaison Recherche vs RL
| Critère | Recherche Classique | RL |
|---------|--------------------|----|
| Connaissance requise | Modèle complet de l'env. | Aucun (apprentissage) |
| Optimalité | Garantie (A*, Dijkstra) | Asymptotique |
| Adaptabilité | Statique | Dynamique |
| Coût computationnel | Par requête | Phase d'entraînement |

---

## Compétences Démontrées

Ce projet mobilise les compétences suivantes :

1. **Formalisation de problèmes** — Modélisation d'un environnement comme un graphe pondéré
2. **Résolution algorithmique** — Implémentation et comparaison de 4 algorithmes de recherche
3. **Prise de décision** — Agents RL apprenant une politique de navigation
4. **Optimisation** — Analyse de complexité, benchmarks systématiques
5. **Analyse critique** — Comparaison qualitative et quantitative des approches

---

## Technologies

- **Python 3.9+** — Langage principal
- **NumPy** — Calcul numérique, tables Q
- **Matplotlib** — Visualisation (grilles, heatmaps, courbes)
- **Streamlit** — Interface web interactive
- **Pandas** — Analyse des benchmarks

---

## Références

- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach*
- Sutton, R. & Barto, A. *Reinforcement Learning: An Introduction*
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A* algorithm
- Watkins, C. J., & Dayan, P. (1992). Q-learning

---

## Auteur

**LIU Qiange**
