from collections.abc import Callable

import numpy as np

from .config import DIM, ITERATIONS, N_PARTICLES
from .pso import PSO


def run_pso(
    func: Callable[[np.ndarray], float],
    bounds: tuple[float, float],
    n_particles: int = N_PARTICLES,
    iterations: int = ITERATIONS,
) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng()
    positions = rng.uniform(bounds[0], bounds[1], size=(n_particles, DIM))

    pso = PSO(positions, func, iterations)

    history = []

    def callback(_it: int, positions: np.ndarray, _gbest: np.ndarray) -> None:
        history.append(positions.copy())

    global_best = pso.optimize(callback=callback)

    return global_best, history
