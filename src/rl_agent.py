"""
Reinforcement Learning Agents for GridWorld.

Implements tabular Q-Learning and SARSA with epsilon-greedy exploration.
Tracks learning curves and converged policies for comparison with
classical search approaches.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np

from .environment import GridWorld


@dataclass
class TrainingResult:
    algorithm: str
    episode_rewards: List[float]
    episode_steps: List[int]
    q_table: np.ndarray
    policy_path: List[Tuple[int, int]]
    total_episodes: int
    converged_at: Optional[int] = None


class QLearningAgent:
    """Tabular Q-Learning with epsilon-greedy exploration."""

    def __init__(
        self,
        env: GridWorld,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        self.n_actions = len(env.ACTIONS)
        self.q_table = np.zeros((env.rows, env.cols, self.n_actions))

    def choose_action(self, state: Tuple[int, int]) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        r, c = state
        return int(np.argmax(self.q_table[r, c]))

    def train(self, episodes: int = 1000, max_steps: int = 500) -> TrainingResult:
        episode_rewards = []
        episode_steps = []
        best_avg = -float("inf")
        converged_at = None

        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0.0
            steps = 0

            for _ in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(state, action)

                r, c = state
                nr, nc = next_state
                best_next = np.max(self.q_table[nr, nc])
                td_target = reward + self.gamma * best_next * (1 - done)
                self.q_table[r, c, action] += self.alpha * (td_target - self.q_table[r, c, action])

                state = next_state
                total_reward += reward
                steps += 1
                if done:
                    break

            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if ep >= 100 and converged_at is None:
                avg = np.mean(episode_rewards[-100:])
                if avg > best_avg:
                    best_avg = avg
                if avg > 50.0:
                    converged_at = ep

        policy_path = self._extract_policy_path()
        return TrainingResult(
            algorithm="Q-Learning",
            episode_rewards=episode_rewards,
            episode_steps=episode_steps,
            q_table=self.q_table.copy(),
            policy_path=policy_path,
            total_episodes=episodes,
            converged_at=converged_at,
        )

    def _extract_policy_path(self, max_steps: int = 500) -> List[Tuple[int, int]]:
        state = self.env.reset()
        path = [state]
        for _ in range(max_steps):
            r, c = state
            action = int(np.argmax(self.q_table[r, c]))
            next_state, _, done = self.env.step(state, action)
            if next_state == state:
                break
            path.append(next_state)
            state = next_state
            if done:
                break
        return path


class SARSAAgent:
    """Tabular SARSA with epsilon-greedy exploration."""

    def __init__(
        self,
        env: GridWorld,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = random.Random(seed)

        self.n_actions = len(env.ACTIONS)
        self.q_table = np.zeros((env.rows, env.cols, self.n_actions))

    def choose_action(self, state: Tuple[int, int]) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        r, c = state
        return int(np.argmax(self.q_table[r, c]))

    def train(self, episodes: int = 1000, max_steps: int = 500) -> TrainingResult:
        episode_rewards = []
        episode_steps = []
        best_avg = -float("inf")
        converged_at = None

        for ep in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            total_reward = 0.0
            steps = 0

            for _ in range(max_steps):
                next_state, reward, done = self.env.step(state, action)
                next_action = self.choose_action(next_state)

                r, c = state
                nr, nc = next_state
                td_target = reward + self.gamma * self.q_table[nr, nc, next_action] * (1 - done)
                self.q_table[r, c, action] += self.alpha * (td_target - self.q_table[r, c, action])

                state = next_state
                action = next_action
                total_reward += reward
                steps += 1
                if done:
                    break

            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if ep >= 100 and converged_at is None:
                avg = np.mean(episode_rewards[-100:])
                if avg > best_avg:
                    best_avg = avg
                if avg > 50.0:
                    converged_at = ep

        policy_path = self._extract_policy_path()
        return TrainingResult(
            algorithm="SARSA",
            episode_rewards=episode_rewards,
            episode_steps=episode_steps,
            q_table=self.q_table.copy(),
            policy_path=policy_path,
            total_episodes=episodes,
            converged_at=converged_at,
        )

    def _extract_policy_path(self, max_steps: int = 500) -> List[Tuple[int, int]]:
        state = self.env.reset()
        path = [state]
        for _ in range(max_steps):
            r, c = state
            action = int(np.argmax(self.q_table[r, c]))
            next_state, _, done = self.env.step(state, action)
            if next_state == state:
                break
            path.append(next_state)
            state = next_state
            if done:
                break
        return path
