"""
Classical Pathfinding Algorithms.

Implements BFS, DFS, Dijkstra, and A* on a GridWorld,
returning the path, visited nodes, and performance metrics.
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

from .environment import GridWorld


@dataclass
class SearchResult:
    algorithm: str
    path: List[Tuple[int, int]]
    visited_order: List[Tuple[int, int]]
    path_cost: float
    nodes_explored: int
    time_seconds: float
    found: bool

    @property
    def path_length(self) -> int:
        return len(self.path) if self.path else 0


def _reconstruct(came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ── BFS ─────────────────────────────────────────────────────────
def bfs(env: GridWorld) -> SearchResult:
    t0 = time.perf_counter()
    queue = deque([env.start])
    visited: Set[Tuple[int, int]] = {env.start}
    visited_order: List[Tuple[int, int]] = [env.start]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    while queue:
        current = queue.popleft()
        if current == env.goal:
            path = _reconstruct(came_from, current)
            cost = sum(env.step_cost(r, c) for r, c in path[1:])
            return SearchResult("BFS", path, visited_order, cost,
                                len(visited), time.perf_counter() - t0, True)

        for neighbor in env.neighbors(*current):
            if neighbor not in visited:
                visited.add(neighbor)
                visited_order.append(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return SearchResult("BFS", [], visited_order, 0.0,
                        len(visited), time.perf_counter() - t0, False)


# ── DFS ─────────────────────────────────────────────────────────
def dfs(env: GridWorld) -> SearchResult:
    t0 = time.perf_counter()
    stack = [env.start]
    visited: Set[Tuple[int, int]] = set()
    visited_order: List[Tuple[int, int]] = []
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        visited_order.append(current)

        if current == env.goal:
            path = _reconstruct(came_from, current)
            cost = sum(env.step_cost(r, c) for r, c in path[1:])
            return SearchResult("DFS", path, visited_order, cost,
                                len(visited), time.perf_counter() - t0, True)

        for neighbor in env.neighbors(*current):
            if neighbor not in visited:
                came_from[neighbor] = current
                stack.append(neighbor)

    return SearchResult("DFS", [], visited_order, 0.0,
                        len(visited), time.perf_counter() - t0, False)


# ── Dijkstra ────────────────────────────────────────────────────
def dijkstra(env: GridWorld) -> SearchResult:
    t0 = time.perf_counter()
    dist: Dict[Tuple[int, int], float] = {env.start: 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited_order: List[Tuple[int, int]] = []
    pq = [(0.0, env.start)]
    closed: Set[Tuple[int, int]] = set()

    while pq:
        cost, current = heapq.heappop(pq)
        if current in closed:
            continue
        closed.add(current)
        visited_order.append(current)

        if current == env.goal:
            path = _reconstruct(came_from, current)
            return SearchResult("Dijkstra", path, visited_order, cost,
                                len(closed), time.perf_counter() - t0, True)

        for neighbor in env.neighbors(*current):
            if neighbor in closed:
                continue
            new_cost = cost + env.step_cost(*neighbor)
            if new_cost < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))

    return SearchResult("Dijkstra", [], visited_order, 0.0,
                        len(closed), time.perf_counter() - t0, False)


# ── A* ──────────────────────────────────────────────────────────
def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(env: GridWorld, heuristic=None) -> SearchResult:
    if heuristic is None:
        heuristic = _manhattan

    t0 = time.perf_counter()
    g_score: Dict[Tuple[int, int], float] = {env.start: 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited_order: List[Tuple[int, int]] = []
    f0 = heuristic(env.start, env.goal)
    pq = [(f0, 0.0, env.start)]  # (f, g, node)
    closed: Set[Tuple[int, int]] = set()

    while pq:
        f, g, current = heapq.heappop(pq)
        if current in closed:
            continue
        closed.add(current)
        visited_order.append(current)

        if current == env.goal:
            path = _reconstruct(came_from, current)
            return SearchResult("A*", path, visited_order, g,
                                len(closed), time.perf_counter() - t0, True)

        for neighbor in env.neighbors(*current):
            if neighbor in closed:
                continue
            new_g = g + env.step_cost(*neighbor)
            if new_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = new_g
                f_new = new_g + heuristic(neighbor, env.goal)
                came_from[neighbor] = current
                heapq.heappush(pq, (f_new, new_g, neighbor))

    return SearchResult("A*", [], visited_order, 0.0,
                        len(closed), time.perf_counter() - t0, False)


# ── convenience ─────────────────────────────────────────────────
ALL_ALGORITHMS = {
    "BFS": bfs,
    "DFS": dfs,
    "Dijkstra": dijkstra,
    "A*": astar,
}


def run_all(env: GridWorld) -> List[SearchResult]:
    return [algo(env) for algo in ALL_ALGORITHMS.values()]
