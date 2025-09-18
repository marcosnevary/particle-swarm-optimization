from collections.abc import Callable

import numpy as np

from .config import COGNITIVE, INERTIA, SOCIAL


class PSO:
    def __init__(
        self,
        positions: np.ndarray,
        fitness: Callable[[np.ndarray], float],
        iterations: int,
    ) -> None:
        self.fitness = fitness
        self.iterations = iterations
        self.positions = positions
        self.velocities = np.zeros_like(self.positions)
        self.personal_best = self.positions.copy()

        best_idx = np.argmin([self.fitness(pb) for pb in self.personal_best])
        self.global_best = self.personal_best[best_idx].copy()

    def _update_velocities(
        self,
        inertia: float,
        cognitive: float,
        social: float,
    ) -> None:
        rng = np.random.default_rng()
        r1 = rng.uniform(0, 1, size=self.positions.shape)
        r2 = rng.uniform(0, 1, size=self.positions.shape)

        self.velocities = (
            inertia * self.velocities
            + cognitive * r1 * (self.personal_best - self.positions)
            + social * r2 * (self.global_best - self.positions)
        )

    def _update_positions(self) -> None:
        self.positions += self.velocities

    def _update_best_positions(self) -> None:
        for i in range(self.positions.shape[0]):
            if self.fitness(self.positions[i]) < self.fitness(self.personal_best[i]):
                self.personal_best[i] = self.positions[i].copy()
            if self.fitness(self.positions[i]) < self.fitness(self.global_best):
                self.global_best = self.positions[i].copy()

    def optimize(
        self,
        inertia: float = INERTIA,
        cognitive: float = COGNITIVE,
        social: float = SOCIAL,
        callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
    ) -> np.ndarray:
        for it in range(self.iterations):
            self._update_velocities(inertia, cognitive, social)
            self._update_positions()
            self._update_best_positions()
            if callback:
                callback(it, self.positions, self.global_best)
        return self.global_best
