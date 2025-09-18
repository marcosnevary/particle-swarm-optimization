# 🌀 Particle Swarm Optimization (PSO)

This project implements the **Particle Swarm Optimization (PSO)** algorithm to find the global minimum of various benchmark functions.

## About the Algorithm

### Introduction

**Particle Swarm Optimization (PSO)** is a meta-heuristic optimization algorithm proposed by Kennedy and Eberhart in 1995. It is a population-based method inspired by the collective behavior of animals such as flocks of birds or schools of fish. PSO is particularly useful for solving complex optimization problems where the search space is large or no analytical solution exists.

### Intuition

Imagine a swarm of particles moving through a space in search of the highest or lowest point. Each particle has a position and velocity, and it keeps track of the best position it has found so far. Additionally, particles in the swarm share information about the best position found by the entire swarm.

The position and velocity of each particle are influenced by three main components:

1. **Inertia**: tendency of the particle to maintain its current trajectory.
2. **Cognitive component**: the best solution found by the particle itself.
3. **Social component**: the best solution found by the swarm.

In this context, the swarm represents the population exploring the search space (set of possible solutions), with each particle acting as a candidate solution. The particle's position corresponds to a solution, while its velocity determines how it moves through the space in search of the global optimum.

### Definition

Let $f: S \subseteq \mathbb{R}^d \to \mathbb{R}$ be the objective function, with $d \in \mathbb{N}^+$, where $S$ represents the $d$-dimensional search space. The optimization problem is defined as:
$$\min_{x \in S} f(x)$$

The swarm consists of $n$ particles. Each particle $i \in [1, n]$ at time $t \in \mathbb{N}$ has: a position $x_i^t \in \mathbb{R}^d$, a velocity $v_i^t \in \mathbb{R}^d$, the best personal position $p_i^t \in \mathbb{R}^d$, and the best global position found by the swarm $g^t \in \mathbb{R}^d$.

At each iteration $t$, the velocity and position are updated using:

$$
v_i^{t + 1} = wv_i^{t} + c_1r_{1, i}^t(p_i^t - x_i^t) + c_2r_{2,i}^t(g^t - x_i^t)
$$

where $w > 0$ is the inertia weight, $c_1$ and $c_2$ are cognitive and social coefficients, and $r_{1,i}^t, r_{2,i}^t \sim U(0, 1)$ are random numbers uniformly drawn from $[0, 1]$.

The position is updated as:

$$
x_i^{t + 1} = x_i^t + v_i^{t + 1}
$$

The personal best and global best are updated as:

$$
p_i^{t + 1} =
\begin{cases}
x_i^{t + 1} & \text{if } f(x_i^{t + 1}) < f(p_i^t) \\
p_i^t & \text{otherwise}
\end{cases}
$$

$$
g^{t + 1} =
\begin{cases}
x_i^{t + 1} & \text{if } f(x_i^{t + 1}) < f(g^t) \\
g^t & \text{otherwise}
\end{cases}
$$

## Technologies

- Python 3.13
- NumPy – array manipulation
- matplotlib – visualization and animation

## Project Structure

```
particle-swarm-optimization/
│
├── main.py # Runs PSO on benchmark functions and generates animations
├── src/
│ ├── pso.py # PSO class implementing the core algorithm
│ ├── run_pso.py # Configures and executes PSO, collects particle positions
│ ├── create_animation.py # Generates .gif animations
│ ├── benchmarks.py # Benchmark functions
│ └── config.py # Constants and configuration settings for PSO
└── results/ # Folder where generated animation files (.gif) are saved
```

## How to Run

1. **Clone the repository:**

   ```bash
   git clone https://github.com/marcosnevary/particle-swarm-optimization.git
   cd particle-swarm-optimization

   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

3. **Run:**

   ```bash
   python main.py
   ```
