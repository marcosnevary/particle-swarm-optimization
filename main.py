from src.benchmarks import ackley, himmelblau, rastrigin, rosenbrock, sphere
from src.config import FUNCTIONS_BOUNDS
from src.create_animation import create_animation
from src.run_pso import run_pso

function_map = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "ackley": ackley,
    "himmelblau": himmelblau,
}

for function_name, bounds in FUNCTIONS_BOUNDS:
    function = function_map[function_name]
    print(f"Running {function_name}...")

    global_best, history = run_pso(function, bounds)
    print(f"Result ({function_name}): {global_best}")
    create_animation(function, history, bounds, filename=f"{function_name}.gif")
