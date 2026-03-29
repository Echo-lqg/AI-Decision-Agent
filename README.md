# AI Decision Agent — Planification de Chemins & Apprentissage par Renforcement

> **Projet personnel d'autoformation** — Étude comparative entre algorithmes
> de recherche classiques (BFS, DFS, Dijkstra, A\*) et agents par apprentissage
> par renforcement (Q-Learning, SARSA) pour la navigation autonome dans un
> environnement GridWorld configurable avec terrain pondéré.

---

## Contexte & Objectifs

Ce projet est né d'une curiosité personnelle pour les problèmes de **prise de
décision autonome** — un sujet que je n'avais pas abordé dans mon cursus de M1.
En partant des ouvrages de référence (Russell & Norvig, Sutton & Barto), j'ai
cherché à comprendre et à implémenter par moi-même deux approches fondamentales :

- **La recherche dans les graphes** : comment un agent peut-il trouver un chemin
  optimal lorsqu'il dispose d'une connaissance complète de son environnement ?
- **L'apprentissage par renforcement** : comment un agent peut-il apprendre à
  naviguer *sans* modèle préalable, uniquement par essai-erreur ?

Au-delà de l'implémentation, l'objectif était de **confronter expérimentalement
et rigoureusement** ces deux paradigmes — avec des hypothèses explicites, des
environnements identiques pour chaque comparaison, et des tests de significativité
statistique — pour mieux comprendre leurs forces, leurs limites, et les situations
où l'un est préférable à l'autre.

---

## Description du Projet

### 1. Algorithmes de Recherche Classiques

Quatre algorithmes implémentés from scratch (sans bibliothèque de graphes) :

| Algorithme | Type | Optimalité | Complexité |
|-----------|------|-----------|------------|
| **BFS** | Non-informé | Optimal (coût uniforme) | O(V + E) |
| **DFS** | Non-informé | Non optimal | O(V + E) |
| **Dijkstra** | Informé par coût | Optimal (coûts ≥ 0) | O((V+E) log V) |
| **A\*** | Heuristique (Manhattan) | Optimal (h admissible) | O(E log V) |

### 2. Apprentissage par Renforcement Tabulaire

| Agent | Mise à jour | Propriété |
|-------|-----------|-----------|
| **Q-Learning** | Off-policy : `max_a Q(s',a)` | Converge vers Q\* optimal, exploration agressive |
| **SARSA** | On-policy : `Q(s',a')` | Politiques plus prudentes, pénalités mieux évitées |

### 3. Environnement GridWorld

L'environnement est modélisé comme un graphe pondéré implicite :

- Grilles configurables (taille de 7×7 à 51×51)
- **Terrain pondéré** : cases marécage (coût de traversée ×3)
- **Système de récompenses** : cellules bonus (+10), pénalité de vie (-1), objectif (+100)
- **Génération procédurale de labyrinthes** : DFS récursif et algorithme de Prim

---

## Architecture du Projet

```
AI-Decision-Agent/
├── README.md                 # Documentation
├── requirements.txt          # Dépendances Python
├── main.py                   # Point d'entrée CLI (démonstration, benchmarks)
├── app.py                    # Interface web interactive (Streamlit)
└── src/
    ├── __init__.py
    ├── environment.py        # Modèle GridWorld, dynamique, génération de labyrinthes
    ├── pathfinding.py        # BFS, DFS, Dijkstra, A* — implémentés from scratch
    ├── rl_agent.py           # Q-Learning & SARSA tabulaires, epsilon-greedy
    ├── visualizer.py         # Visualisation : grilles, heatmaps, courbes, value maps
    └── benchmark.py          # Expérimentations systématiques et analyse statistique
```

Principes de conception :
- Séparation entre environnement, algorithmes et visualisation
- Interface commune (`SearchResult`, `TrainingResult`) facilitant la comparaison
- Code modulaire, documenté avec docstrings

---

## Installation & Utilisation

### Prérequis
- Python 3.9+

### Installation
```bash
git clone https://github.com/Echo-lqg/AI-Decision-Agent.git
cd AI-Decision-Agent
pip install -r requirements.txt
```

### CLI — Ligne de Commande
```bash
# Démonstration complète (pathfinding + RL + visualisations)
python main.py demo

# Comparaison des algorithmes de recherche
python main.py pathfinding --size 21 --seed 42

# Pathfinding dans un labyrinthe généré
python main.py pathfinding --size 21 --maze dfs

# Entraînement des agents RL
python main.py rl --size 11 --episodes 2000

# Benchmarks systématiques
python main.py benchmark --type pathfinding
python main.py benchmark --type maze
python main.py benchmark --type rl

# Comparaison croisée (search vs RL, mêmes grilles) + tests statistiques
python main.py benchmark --type cross
```

### Interface Web Interactive
```bash
streamlit run app.py
```

L'interface Streamlit permet de :
- Configurer l'environnement en temps réel (taille, densité d'obstacles, type de labyrinthe)
- Exécuter et comparer visuellement les algorithmes de recherche
- Entraîner les agents RL avec des hyperparamètres ajustables (α, γ, ε-decay)
- Lancer des benchmarks et consulter les résultats sous forme de tableaux et graphiques

---

## Protocole Expérimental & Hypothèses

### Hypothèses

| # | Hypothèse | Métrique | Test |
|---|-----------|----------|------|
| **H1** | A\* explore moins de nœuds que BFS (guidance heuristique) | `nodes_explored` | Wilcoxon signé |
| **H2** | Dijkstra trouve des chemins de moindre coût que BFS sur terrain pondéré | `path_cost` | Wilcoxon signé |
| **H3** | Q-Learning converge plus vite que SARSA | `converged_at` | Wilcoxon signé |
| **H4** | SARSA produit des chemins de coût inférieur ou égal à Q-Learning | `path_cost` | Wilcoxon signé |

### Protocole

- **Environnements identiques** : chaque comparaison (pathfinding vs pathfinding, RL vs RL,
  et surtout pathfinding vs RL) est effectuée sur **exactement la même grille générée**,
  incluant obstacles et terrain pondéré (marécages, coût = 3).
- **Données appariées** : chaque trial produit une paire d'observations (même grille →
  résultat algo A et résultat algo B), ce qui justifie un test **apparié**.
- **Répétitions multiples** : chaque configuration est testée sur *n* essais indépendants
  (seeds différents) pour réduire la variance.
- **Asymétrie assumée** : la comparaison croisée (recherche vs RL) met en évidence des
  différences de performance, mais reste **intrinsèquement asymétrique** : les algorithmes
  de recherche classiques disposent d'une connaissance complète de l'environnement
  (*full model*), tandis que l'apprentissage par renforcement opère sans modèle
  (*model-free*) et doit découvrir la structure par exploration. Cette distinction est
  fondamentale en intelligence artificielle et les résultats doivent être interprétés
  en tenant compte de cette différence de paradigme.

### Analyse statistique

Nous utilisons le **test de Wilcoxon signé** (*signed-rank*, non-paramétrique) pour évaluer
la significativité statistique des différences observées entre algorithmes (α = 0.05).

Ce choix se justifie par :
1. **Plan apparié** — les observations sont appariées par condition d'environnement
   (trial × grid_size × obstacle_ratio), ce qui exclut les tests pour échantillons
   indépendants (Mann-Whitney U, t-test indépendant).
2. **Distribution non-normale** — les métriques (nœuds explorés, coût du chemin,
   épisode de convergence) ne suivent pas nécessairement une loi normale,
   ce qui exclut le t-test apparié paramétrique.
3. **Taille d'effet** — en complément de la p-value, nous rapportons *r* = |Z| / √N
   pour quantifier l'ampleur pratique de la différence, indépendamment de la taille
   de l'échantillon. Le Z est obtenu directement via l'approximation normale de SciPy
   (`method="approx"`) qui intègre la correction pour les ex-æquo (*ties*).

```bash
# Lancer la comparaison croisée avec tests statistiques
python main.py benchmark --type cross
```

---

## Résultats & Observations

### Recherche Classique

- **A\*** explore significativement moins de nœuds que BFS grâce à l'heuristique Manhattan, tout en conservant l'optimalité.
- **Dijkstra** s'avère indispensable lorsque le terrain est pondéré (marécages), là où BFS ne garantit plus l'optimalité.
- **DFS** est rapide mais produit des chemins souvent bien plus longs que l'optimal.
- L'écart de performance entre A\* et BFS s'accentue avec la taille de la grille.

### Apprentissage par Renforcement

- **Q-Learning** (off-policy) tend à converger plus vite vers une politique efficace, mais peut emprunter des trajectoires risquées.
- **SARSA** (on-policy) apprend des politiques plus prudentes, évitant davantage les zones à forte pénalité.
- La convergence des deux agents nécessite un nombre suffisant d'épisodes (~500-1000 selon la complexité de la grille).

### Synthèse : Recherche Classique vs RL

| Critère | Recherche Classique | Apprentissage par Renforcement |
|---------|--------------------|---------------------------------|
| **Modèle requis** | Complet (graphe explicite) | Aucun (apprentissage par essai-erreur) |
| **Optimalité** | Garantie (A\*, Dijkstra) | Asymptotique (convergence sous conditions) |
| **Adaptabilité** | Recalcul si l'environnement change | Adaptation par ré-entraînement |
| **Coût** | Par requête (temps réel) | Phase d'entraînement initiale |

> Les résultats détaillés (p-values, tailles d'effet) sont générés automatiquement
> par `python main.py benchmark --type cross` et sauvegardés dans `output_statistical_tests.csv`.

---

## Ce Que Ce Projet M'a Apporté

Ce projet, réalisé en autodidacte en dehors de mon cursus de M1, m'a permis de :

- **Découvrir concrètement** les algorithmes de recherche dans les graphes en les implémentant from scratch, et comprendre pourquoi A\* est si largement utilisé en pratique.
- **Aborder l'apprentissage par renforcement** à travers les cas tabulaires (Q-Learning, SARSA), et saisir la distinction fondamentale entre approches on-policy et off-policy.
- **Formaliser un problème** en le modélisant comme un MDP (états, actions, transitions, récompenses).
- **Développer une démarche expérimentale** : benchmarks systématiques, analyse statistique, comparaison quantitative entre approches.
- **Faire le lien entre mes acquis en psychologie et les modèles computationnels de prise de décision** : les notions de récompense, d'exploration/exploitation et d'apprentissage par essai-erreur trouvent un écho direct dans les théories comportementales que j'ai étudiées, et ce projet m'a permis de les formaliser mathématiquement.
- **Identifier mes limites** : ce projet reste dans le cadre tabulaire ; j'aimerais explorer par la suite les méthodes avec approximation de fonctions (Deep RL) et les environnements plus complexes.

---

## Technologies

| Technologie | Rôle |
|-------------|------|
| **Python 3.9+** | Langage principal |
| **NumPy** | Calcul numérique, tables Q |
| **Matplotlib** | Visualisation (grilles, heatmaps, courbes d'entraînement, value maps) |
| **Streamlit** | Interface web interactive |
| **Pandas** | Analyse statistique des benchmarks |
| **SciPy** | Tests de significativité statistique (Mann-Whitney U) |

---

## Références

- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach* (4th ed., 2020)
- Sutton, R. & Barto, A. *Reinforcement Learning: An Introduction* (2nd ed., 2018)
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*. IEEE Transactions on Systems Science and Cybernetics.
- Watkins, C. J. & Dayan, P. (1992). *Q-Learning*. Machine Learning, 8(3-4), 279-292.
- Rummery, G. A. & Niranjan, M. (1994). *On-Line Q-Learning Using Connectionist Systems*. Technical Report CUED/F-INFENG/TR 166, Cambridge University.

---

## Auteur

**LIU Qiange**
