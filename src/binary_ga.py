"""Binary-encoded genetic algorithm implemented from scratch."""

from typing import Callable, Optional

import numpy as np

from .stopping import StagnationConfig, initialize_stagnation, update_stagnation


def bits_per_variable(bounds: np.ndarray, precision_digits: int = 6) -> np.ndarray:
    """Compute the number of bits needed for each variable."""
    bounds = np.asarray(bounds, dtype=float)
    ranges = bounds[:, 1] - bounds[:, 0]
    return np.ceil(np.log2(ranges * (10**precision_digits) + 1.0)).astype(int)


def decode_population(population: np.ndarray, bounds: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """Decode a binary population into real-valued vectors."""
    population = np.asarray(population, dtype=np.int8)
    bounds = np.asarray(bounds, dtype=float)
    decoded = np.empty((population.shape[0], len(bits)), dtype=float)
    start = 0
    for variable_index, bit_count in enumerate(bits):
        end = start + int(bit_count)
        segment = population[:, start:end].astype(np.int64)
        powers = (2 ** np.arange(bit_count - 1, -1, -1, dtype=np.int64)).reshape(-1, 1)
        integer_values = segment @ powers
        integer_values = integer_values.reshape(-1).astype(float)
        lower, upper = bounds[variable_index]
        decoded[:, variable_index] = lower + integer_values * (upper - lower) / (2**bit_count - 1)
        start = end
    return np.clip(decoded, bounds[:, 0], bounds[:, 1])


def _evaluate(objective: Callable[[np.ndarray], np.ndarray], decoded_population: np.ndarray) -> np.ndarray:
    fitness = objective(decoded_population)
    return np.asarray(fitness, dtype=float).reshape(-1)


def _roulette_select(population: np.ndarray, fitness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    min_fitness = float(np.min(fitness))
    shifted = fitness - min_fitness
    if np.all(shifted <= 1e-14):
        probabilities = np.full(population.shape[0], 1.0 / population.shape[0])
    else:
        scores = 1.0 / (shifted + 1e-12)
        if (not np.all(np.isfinite(scores))) or float(np.sum(scores)) <= 0.0:
            probabilities = np.full(population.shape[0], 1.0 / population.shape[0])
        else:
            probabilities = scores / np.sum(scores)
    index = rng.choice(population.shape[0], p=probabilities)
    return population[index].copy()


def _single_point_crossover(parent_1: np.ndarray, parent_2: np.ndarray, pc: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    child_1 = parent_1.copy()
    child_2 = parent_2.copy()
    if parent_1.size > 1 and rng.random() < pc:
        point = int(rng.integers(1, parent_1.size))
        child_1 = np.concatenate((parent_1[:point], parent_2[point:])).astype(np.int8)
        child_2 = np.concatenate((parent_2[:point], parent_1[point:])).astype(np.int8)
    return child_1, child_2


def _bit_flip_mutation(chromosome: np.ndarray, mutation_probability: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(chromosome.size) < mutation_probability
    mutated = chromosome.copy()
    mutated[mask] = 1 - mutated[mask]
    return mutated


def run_binary_ga(
    objective: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    population_size: int = 100,
    pc: float = 0.9,
    elite_size: int = 1,
    max_generations: int = 1000,
    stopping_config: Optional[StagnationConfig] = None,
    seed: Optional[int] = None,
    precision_digits: int = 6,
) -> dict[str, object]:
    """Run a binary-encoded minimization GA."""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    bits = bits_per_variable(bounds, precision_digits=precision_digits)
    chromosome_length = int(np.sum(bits))
    mutation_probability = 1.0 / chromosome_length
    elite_size = max(0, min(int(elite_size), population_size))

    if stopping_config is not None:
        max_generations = stopping_config.max_generations

    population = rng.integers(0, 2, size=(population_size, chromosome_length), dtype=np.int8)
    decoded = decode_population(population, bounds, bits)
    fitness = _evaluate(objective, decoded)

    best_index = int(np.argmin(fitness))
    best_fitness = float(fitness[best_index])
    best_solution = decoded[best_index].copy()
    best_history = [best_fitness]
    stagnation_state = initialize_stagnation(best_fitness)
    stopped_generation = 0

    for generation in range(1, max_generations + 1):
        next_population = []

        if elite_size > 0:
            elite_indices = np.argsort(fitness)[:elite_size]
            next_population.extend(population[elite_indices].copy())

        while len(next_population) < population_size:
            parent_1 = _roulette_select(population, fitness, rng)
            parent_2 = _roulette_select(population, fitness, rng)
            child_1, child_2 = _single_point_crossover(parent_1, parent_2, pc, rng)
            child_1 = _bit_flip_mutation(child_1, mutation_probability, rng)
            child_2 = _bit_flip_mutation(child_2, mutation_probability, rng)
            next_population.append(child_1)
            if len(next_population) < population_size:
                next_population.append(child_2)

        population = np.asarray(next_population[:population_size], dtype=np.int8)
        decoded = decode_population(population, bounds, bits)
        fitness = _evaluate(objective, decoded)

        generation_best_index = int(np.argmin(fitness))
        generation_best_fitness = float(fitness[generation_best_index])
        generation_best_solution = decoded[generation_best_index].copy()
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
            "encoding": "binary",
            "population_size": int(population_size),
            "pc": float(pc),
            "pm": float(mutation_probability),
            "elite_size": int(elite_size),
            "bits_per_variable": bits.astype(int).tolist(),
            "chromosome_length": int(chromosome_length),
            "precision_digits": int(precision_digits),
            "seed": None if seed is None else int(seed),
            "function_evaluations": int(population_size * (stopped_generation + 1)),
        },
    }

