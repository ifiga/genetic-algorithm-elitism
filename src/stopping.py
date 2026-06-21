"""Automatic stopping criteria for the genetic algorithms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StagnationConfig:
    """Stagnation-based automatic stopping criterion."""

    min_generations: int = 100
    max_generations: int = 1000
    patience: int = 50
    tol: float = 1e-8


@dataclass
class StagnationState:
    """Mutable state used while checking the stopping criterion."""

    best_reference: float
    last_improvement_generation: int = 0


def initialize_stagnation(best_fitness: float) -> StagnationState:
    return StagnationState(best_reference=float(best_fitness), last_improvement_generation=0)


def update_stagnation(state: StagnationState, generation: int, best_fitness: float, config: StagnationConfig) -> bool:
    """Return True when the run should stop."""
    if best_fitness < state.best_reference - config.tol:
        state.best_reference = float(best_fitness)
        state.last_improvement_generation = int(generation)
        return False

    if generation < config.min_generations:
        return False

    return (generation - state.last_improvement_generation) >= config.patience

