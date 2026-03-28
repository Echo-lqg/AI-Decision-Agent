"""
Visualization module using matplotlib.

Provides rich visual representations of:
- Grid worlds with terrain types
- Algorithm exploration patterns (heatmaps)
- Path overlays
- RL training curves & value maps
- Side-by-side algorithm comparison
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

from .environment import GridWorld, CellType
from .pathfinding import SearchResult
from .rl_agent import TrainingResult


CELL_COLORS = {
    CellType.EMPTY: "#FFFFFF",
    CellType.WALL: "#2C3E50",
    CellType.START: "#2ECC71",
    CellType.GOAL: "#E74C3C",
    CellType.SWAMP: "#8E6B3D",
    CellType.REWARD: "#F39C12",
}


def _draw_grid(ax, env: GridWorld, title: str = ""):
    """Draw the base grid on an axes."""
    grid_rgb = np.ones((env.rows, env.cols, 3))
    for r in range(env.rows):
        for c in range(env.cols):
            color = mcolors.to_rgb(CELL_COLORS[CellType(env.grid[r, c])])
            grid_rgb[r, c] = color
    ax.imshow(grid_rgb, interpolation="nearest")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_grid(env: GridWorld, title: str = "GridWorld", save_path: Optional[str] = None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _draw_grid(ax, env, title)

    legend_elements = [
        mpatches.Patch(facecolor=CELL_COLORS[CellType.START], label="Start"),
        mpatches.Patch(facecolor=CELL_COLORS[CellType.GOAL], label="Goal"),
        mpatches.Patch(facecolor=CELL_COLORS[CellType.WALL], label="Wall"),
        mpatches.Patch(facecolor=CELL_COLORS[CellType.SWAMP], label="Swamp (cost×3)"),
        mpatches.Patch(facecolor=CELL_COLORS[CellType.REWARD], label="Reward (+10)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_search_result(
    env: GridWorld,
    result: SearchResult,
    show_visited: bool = True,
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _draw_grid(ax, env)

    if show_visited and result.visited_order:
        for i, (r, c) in enumerate(result.visited_order):
            alpha = 0.1 + 0.5 * (i / max(len(result.visited_order) - 1, 1))
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor="#3498DB", alpha=min(alpha, 0.6)))

    if result.path:
        path_r = [p[0] for p in result.path]
        path_c = [p[1] for p in result.path]
        ax.plot(path_c, path_r, "o-", color="#E74C3C", linewidth=2.5,
                markersize=5, zorder=5, label="Path")

    stats = (f"{result.algorithm}  |  "
             f"Path: {result.path_length}  |  "
             f"Cost: {result.path_cost:.1f}  |  "
             f"Explored: {result.nodes_explored}  |  "
             f"Time: {result.time_seconds*1000:.2f}ms")
    ax.set_title(stats, fontsize=10, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_comparison(
    env: GridWorld,
    results: List[SearchResult],
    save_path: Optional[str] = None,
):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        _draw_grid(ax, env)

        if result.visited_order:
            for i, (r, c) in enumerate(result.visited_order):
                alpha = 0.1 + 0.5 * (i / max(len(result.visited_order) - 1, 1))
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor="#3498DB", alpha=min(alpha, 0.6)))

        if result.path:
            path_r = [p[0] for p in result.path]
            path_c = [p[1] for p in result.path]
            ax.plot(path_c, path_r, "o-", color="#E74C3C", linewidth=2, markersize=4, zorder=5)

        title = (f"{result.algorithm}\n"
                 f"Path={result.path_length}  Cost={result.path_cost:.1f}\n"
                 f"Explored={result.nodes_explored}  {result.time_seconds*1000:.2f}ms")
        ax.set_title(title, fontsize=9, fontweight="bold")

    fig.suptitle("Algorithm Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_exploration_heatmap(
    env: GridWorld,
    results: List[SearchResult],
    save_path: Optional[str] = None,
):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        heatmap = np.zeros((env.rows, env.cols))
        for i, (r, c) in enumerate(result.visited_order):
            heatmap[r, c] = i + 1

        for r in range(env.rows):
            for c in range(env.cols):
                if env.grid[r, c] == CellType.WALL:
                    heatmap[r, c] = -1

        masked = np.ma.masked_where(heatmap == 0, heatmap)
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color="white")
        im = ax.imshow(masked, cmap=cmap, interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Visit order")

        for r in range(env.rows):
            for c in range(env.cols):
                if env.grid[r, c] == CellType.WALL:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#2C3E50"))

        ax.set_title(f"{result.algorithm} Exploration", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── RL visualizations ───────────────────────────────────────────

def plot_training_curves(
    results: List[TrainingResult],
    window: int = 50,
    save_path: Optional[str] = None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for result in results:
        rewards = np.array(result.episode_rewards)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(smoothed, label=result.algorithm, linewidth=1.5)
        if result.converged_at is not None:
            ax1.axvline(x=result.converged_at, linestyle="--", alpha=0.5)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward (smoothed)")
    ax1.set_title("Training Reward Curves", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for result in results:
        steps = np.array(result.episode_steps)
        smoothed = np.convolve(steps, np.ones(window) / window, mode="valid")
        ax2.plot(smoothed, label=result.algorithm, linewidth=1.5)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps per Episode")
    ax2.set_title("Steps to Goal", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_value_map(
    env: GridWorld,
    q_table: np.ndarray,
    title: str = "State Value Map",
    save_path: Optional[str] = None,
):
    values = np.max(q_table, axis=2)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for r in range(env.rows):
        for c in range(env.cols):
            if env.grid[r, c] == CellType.WALL:
                values[r, c] = np.nan

    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="V(s) = max_a Q(s,a)")

    for r in range(env.rows):
        for c in range(env.cols):
            if env.grid[r, c] == CellType.WALL:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#2C3E50"))
            else:
                best_a = int(np.argmax(q_table[r, c]))
                dr, dc = env.ACTIONS[best_a]
                if values[r, c] is not np.ma.masked:
                    ax.annotate("", xy=(c + dc * 0.3, r + dr * 0.3),
                                xytext=(c, r),
                                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rl_path_on_grid(
    env: GridWorld,
    training_result: TrainingResult,
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _draw_grid(ax, env)

    path = training_result.policy_path
    if path:
        path_r = [p[0] for p in path]
        path_c = [p[1] for p in path]
        ax.plot(path_c, path_r, "o-", color="#9B59B6", linewidth=2.5,
                markersize=5, zorder=5, label=f"{training_result.algorithm} policy")
        ax.legend(fontsize=10)

    reached = path[-1] == env.goal if path else False
    status = "Reached Goal" if reached else "Did NOT reach Goal"
    ax.set_title(f"{training_result.algorithm} Learned Policy — {status}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
