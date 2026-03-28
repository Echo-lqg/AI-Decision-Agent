"""
GridWorld Environment for Agent Navigation & Decision Making.

Supports configurable grid sizes, obstacles, weighted terrain,
rewards, and penalties. Compatible with both classical search
and reinforcement learning agents.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Optional, Dict

import numpy as np


class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    START = 2
    GOAL = 3
    SWAMP = 4   # high traversal cost
    REWARD = 5  # bonus reward cell


@dataclass
class GridWorld:
    rows: int
    cols: int
    grid: np.ndarray = field(init=False)
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = field(default=None)
    rewards: Dict[Tuple[int, int], float] = field(default_factory=dict)

    ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
    ACTION_NAMES = ["right", "left", "down", "up"]

    def __post_init__(self):
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        if self.goal is None:
            self.goal = (self.rows - 1, self.cols - 1)
        self.grid[self.start] = CellType.START
        self.grid[self.goal] = CellType.GOAL

    # ── cost model ──────────────────────────────────────────────
    def step_cost(self, r: int, c: int) -> float:
        cell = self.grid[r, c]
        if cell == CellType.SWAMP:
            return 3.0
        return 1.0

    def reward_at(self, r: int, c: int) -> float:
        if (r, c) == self.goal:
            return 100.0
        if (r, c) in self.rewards:
            return self.rewards[(r, c)]
        cell = self.grid[r, c]
        if cell == CellType.WALL:
            return -10.0
        if cell == CellType.SWAMP:
            return -2.0
        return -1.0  # living penalty

    # ── dynamics ────────────────────────────────────────────────
    def is_valid(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] != CellType.WALL

    def neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        result = []
        for dr, dc in self.ACTIONS:
            nr, nc = r + dr, c + dc
            if self.is_valid(nr, nc):
                result.append((nr, nc))
        return result

    def step(self, state: Tuple[int, int], action_idx: int) -> Tuple[Tuple[int, int], float, bool]:
        """Take an action, return (next_state, reward, done)."""
        dr, dc = self.ACTIONS[action_idx]
        nr, nc = state[0] + dr, state[1] + dc
        if not self.is_valid(nr, nc):
            return state, -5.0, False  # wall bump penalty
        reward = self.reward_at(nr, nc)
        done = (nr, nc) == self.goal
        return (nr, nc), reward, done

    # ── environment generation ──────────────────────────────────
    def add_random_obstacles(self, ratio: float = 0.2, seed: Optional[int] = None):
        rng = random.Random(seed)
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) == self.start or (r, c) == self.goal:
                    continue
                if rng.random() < ratio:
                    self.grid[r, c] = CellType.WALL

    def add_random_swamps(self, ratio: float = 0.1, seed: Optional[int] = None):
        rng = random.Random(seed)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] != CellType.EMPTY:
                    continue
                if rng.random() < ratio:
                    self.grid[r, c] = CellType.SWAMP

    def add_random_rewards(self, count: int = 3, seed: Optional[int] = None):
        rng = random.Random(seed)
        placed = 0
        while placed < count:
            r, c = rng.randint(0, self.rows - 1), rng.randint(0, self.cols - 1)
            if self.grid[r, c] == CellType.EMPTY:
                self.grid[r, c] = CellType.REWARD
                self.rewards[(r, c)] = 10.0
                placed += 1

    def reset(self) -> Tuple[int, int]:
        return self.start

    def copy(self) -> "GridWorld":
        env = GridWorld(self.rows, self.cols, start=self.start, goal=self.goal)
        env.grid = self.grid.copy()
        env.rewards = dict(self.rewards)
        return env


# ── Maze generation ─────────────────────────────────────────────
def generate_maze_dfs(rows: int, cols: int, seed: Optional[int] = None) -> GridWorld:
    """Generate a maze using randomized DFS (recursive backtracking)."""
    rng = random.Random(seed)
    grid = np.ones((rows, cols), dtype=int)  # all walls

    def carve(r, c):
        grid[r, c] = CellType.EMPTY
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        rng.shuffle(directions)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == CellType.WALL:
                grid[r + dr // 2, c + dc // 2] = CellType.EMPTY
                carve(nr, nc)

    carve(0, 0)

    # ensure goal reachable
    goal_r, goal_c = rows - 1, cols - 1
    if rows % 2 == 0:
        goal_r = rows - 2
    if cols % 2 == 0:
        goal_c = cols - 2

    env = GridWorld(rows, cols, start=(0, 0), goal=(goal_r, goal_c))
    env.grid = grid
    env.grid[0, 0] = CellType.START
    env.grid[goal_r, goal_c] = CellType.GOAL
    return env


def generate_maze_prim(rows: int, cols: int, seed: Optional[int] = None) -> GridWorld:
    """Generate a maze using randomized Prim's algorithm."""
    rng = random.Random(seed)
    grid = np.ones((rows, cols), dtype=int)

    start = (0, 0)
    grid[start] = CellType.EMPTY
    walls = []

    def add_walls(r, c):
        for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                walls.append((r + dr // 2, c + dc // 2, nr, nc))

    add_walls(0, 0)

    while walls:
        idx = rng.randint(0, len(walls) - 1)
        wr, wc, nr, nc = walls.pop(idx)
        if grid[nr, nc] == CellType.WALL:
            grid[wr, wc] = CellType.EMPTY
            grid[nr, nc] = CellType.EMPTY
            add_walls(nr, nc)

    goal_r, goal_c = rows - 1, cols - 1
    if rows % 2 == 0:
        goal_r = rows - 2
    if cols % 2 == 0:
        goal_c = cols - 2

    env = GridWorld(rows, cols, start=(0, 0), goal=(goal_r, goal_c))
    env.grid = grid
    env.grid[0, 0] = CellType.START
    env.grid[goal_r, goal_c] = CellType.GOAL
    return env
