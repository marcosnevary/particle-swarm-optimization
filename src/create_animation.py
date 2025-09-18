from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def create_animation(
    func: Callable,
    history: list[np.ndarray],
    bounds: tuple[int, int],
    filename: str,
) -> None:
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    plt.close()

    x = np.linspace(bounds[0], bounds[1], 200)
    y = np.linspace(bounds[0], bounds[1], 200)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.array(
        [
            [func(np.array([x, y])) for x, y in zip(row_x, row_y, strict=False)]
            for row_x, row_y in zip(x_grid, y_grid, strict=False)
        ],
    )

    ax.contour(
        x_grid,
        y_grid,
        z_grid,
        levels=20,
        cmap="viridis",
        linewidths=1.2,
    )

    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_facecolor("white")

    frames = []

    for it, positions in enumerate(history):
        scat = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c="#FF8800",
            s=40,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )

        title_artist = ax.text(
            0.5,
            1.03,
            f"Optimization Path ({filename[:-4]}) - Iteration: {it + 1}",
            transform=ax.transAxes,
            ha="center",
            fontsize=14,
            color="black",
        )

        frames.append([scat, title_artist])

    ani = animation.ArtistAnimation(
        fig,
        frames,
        interval=150,
        blit=True,
        repeat_delay=1000,
    )
    ani.save(f"./results/{filename}", writer="pillow")
    plt.close()
