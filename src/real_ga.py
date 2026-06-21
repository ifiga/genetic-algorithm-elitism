"""Real-valued genetic algorithm implemented from scratch."""

from typing import Callable, Optional

import numpy as np

from .stopping import StagnationConfig, initialize_stagnation, update_stagnation


def _evaluate(objective: Callable[[np.ndarray], np.ndarray], population: np.ndarray) -> np.ndarray:
    fitness = objective(population)
    return np.asarray(fitness, dtype=float).reshape(-1)


def _tournament_select(population: np.ndarray, fitness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    first, second = rng.integers(0, population.shape[0], size=2)
    winner = first if fitness[first] <= fitness[second] else second
    return population[winner].copy()


def _sbx_crossover(
    parent_1: np.ndarray,
    parent_2: np.ndarray,
    bounds: np.ndarray,
    pc: float,
    eta_c: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    child_1 = parent_1.copy()
    child_2 = parent_2.copy()
    if rng.random() >= pc:
        return child_1, child_2

    for variable_index in range(parent_1.size):
        if rng.random() > 0.5:
            continue

        y1 = min(parent_1[variable_index], parent_2[variable_index])
        y2 = max(parent_1[variable_index], parent_2[variable_index])
        lower, upper = bounds[variable_index]

        if abs(y1 - y2) <= 1e-14:
            continue

        rand = rng.random()
        beta = 1.0 + (2.0 * (y1 - lower) / (y2 - y1))
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if rand <= 1.0 / alpha:
            beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
        c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))

        beta = 1.0 + (2.0 * (upper - y2) / (y2 - y1))
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if rand <= 1.0 / alpha:
            beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
        c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

        c1 = float(np.clip(c1, lower, upper))
        c2 = float(np.clip(c2, lower, upper))
        if rng.random() <= 0.5:
            child_1[variable_index] = c2
            child_2[variable_index] = c1
        else:
            child_1[variable_index] = c1
            child_2[variable_index] = c2

    return np.clip(child_1, bounds[:, 0], bounds[:, 1]), np.clip(child_2, bounds[:, 0], bounds[:, 1])


def _polynomial_mutation(
    individual: np.ndarray,
    bounds: np.ndarray,
    mutation_probability: float,
    eta_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mutant = individual.copy()
    for variable_index in range(mutant.size):
        if rng.random() >= mutation_probability:
            continue

        lower, upper = bounds[variable_index]
        if upper <= lower:
            continue

        value = mutant[variable_index]
        delta_1 = (value - lower) / (upper - lower)
        delta_2 = (upper - value) / (upper - lower)
        rand = rng.random()
        mutation_power = 1.0 / (eta_m + 1.0)

        if rand <= 0.5:
            xy = 1.0 - delta_1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
            delta_q = val**mutation_power - 1.0
        else:
            xy = 1.0 - delta_2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
            delta_q = 1.0 - val**mutation_power

        value = value + delta_q * (upper - lower)
        mutant[variable_index] = np.clip(value, lower, upper)

    return np.clip(mutant, bounds[:, 0], bounds[:, 1])


def run_real_ga(
    objective: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    population_size: int = 100,
    pc: float = 0.9,
    elite_size: int = 1,
    max_generations: int = 1000,
    stopping_config: Optional[StagnationConfig] = None,
    seed: Optional[int] = None,
    eta_c: float = 20.0,
    eta_m: float = 20.0,
) -> dict[str, object]:
    """Run a real-valued minimization GA."""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    dimension = bounds.shape[0]
    mutation_probability = 1.0 / dimension
    elite_size = max(0, min(int(elite_size), population_size))

    if stopping_config is not None:
        max_generations = stopping_config.max_generations

    lower = bounds[:, 0]
    upper = bounds[:, 1]
    population = rng.uniform(lower, upper, size=(population_size, dimension))
    fitness = _evaluate(objective, population)

    best_index = int(np.argmin(fitness))
    best_fitness = float(fitness[best_index])
    best_solution = population[best_index].copy()
    best_history = [best_fitness]
    stagnation_state = initialize_stagnation(best_fitness)
    stopped_generation = 0

    for generation in range(1, max_generations + 1):
        next_population = []

        if elite_size > 0:
            elite_indices = np.argsort(fitness)[:elite_size]
            next_population.extend(population[elite_indices].copy())

        while len(next_population) < population_size:
            parent_1 = _tournament_select(population, fitness, rng)
            parent_2 = _tournament_select(population, fitness, rng)
            child_1, child_2 = _sbx_crossover(parent_1, parent_2, bounds, pc, eta_c, rng)
            child_1 = _polynomial_mutation(child_1, bounds, mutation_probability, eta_m, rng)
            child_2 = _polynomial_mutation(child_2, bounds, mutation_probability, eta_m, rng)
            next_population.append(child_1)
            if len(next_population) < population_size:
                next_population.append(child_2)

        population = np.asarray(next_population[:population_size], dtype=float)
        population = np.clip(population, lower, upper)
        fitness = _evaluate(objective, population)

        generation_best_index = int(np.argmin(fitness))
        generation_best_fitness = float(fitness[generation_best_index])
        generation_best_solution = population[generation_best_index].copy()
        best_history.append(generation_best_fitness)

        if generation_best_fitness < best_fitness:
            best_fitness = generation_best_fitness
            best_solution = generation_best_solution

        stopped_generation = generation
        if stopping_config is not None and update_stagnation(stagnation_state, generation, best_fitness, stopping_config):
            break

    return {
        "best_solution": best_solution,
        "best_fitness": float(best_fitness),
        "best_fitness_history": np.asarray(best_history, dtype=float),
        "stopped_generation": int(stopped_generation),
        "metadata": {
            "encoding": "real",
            "population_size": int(population_size),
            "pc": float(pc),
            "pm": float(mutation_probability),
            "elite_size": int(elite_size),
            "eta_c": float(eta_c),
            "eta_m": float(eta_m),
            "dimension": int(dimension),
            "seed": None if seed is None else int(seed),
            "function_evaluations": int(population_size * (stopped_generation + 1)),
        },
    }

