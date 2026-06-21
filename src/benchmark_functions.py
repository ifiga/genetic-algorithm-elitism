"""Benchmark functions used by the genetic algorithm experiments."""

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


@dataclass(frozen=True)
class BenchmarkProblem:
    """Container for a minimization benchmark problem."""

    key: str
    name: str
    dimension: int
    bounds: np.ndarray
    optimum: np.ndarray
    optimum_value: float
    function: Callable[[np.ndarray], np.ndarray]


def _as_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(x, dtype=float)
    was_1d = values.ndim == 1
    if was_1d:
        values = values.reshape(1, -1)
    return values, was_1d


def _return_shape(values: np.ndarray, was_1d: bool) -> np.ndarray | float:
    if was_1d:
        return float(values[0])
    return values


def test_problem_1(x: np.ndarray) -> np.ndarray | float:
    """Rosenbrock-style two-dimensional test problem."""
    values, was_1d = _as_2d(x)
    x1 = values[:, 0]
    x2 = values[:, 1]
    result = 100.0 * (x1**2 - x2) ** 2 + (1.0 - x1) ** 2
    return _return_shape(result, was_1d)


def rastrigin(x: np.ndarray, a: float = 10.0) -> np.ndarray | float:
    """Rastrigin benchmark function."""
    values, was_1d = _as_2d(x)
    n = values.shape[1]
    result = a * n + np.sum(values**2 - a * np.cos(2.0 * np.pi * values), axis=1)
    return _return_shape(result, was_1d)


def ackley(x: np.ndarray, a: float = 20.0, b: float = 1.0 / 5.0, c: float = 2.0 * np.pi) -> np.ndarray | float:
    """Ackley benchmark function."""
    values, was_1d = _as_2d(x)
    n = values.shape[1]
    squared_mean = np.sum(values**2, axis=1) / n
    cosine_mean = np.sum(np.cos(c * values), axis=1) / n
    result = -a * np.exp(-b * np.sqrt(squared_mean)) - np.exp(cosine_mean) + a + np.e
    result = np.maximum(result, 0.0)
    return _return_shape(result, was_1d)


def get_benchmark_problems() -> Dict[str, BenchmarkProblem]:
    """Return the benchmark problems in the assignment."""
    return {
        "test_problem_1": BenchmarkProblem(
            key="test_problem_1",
            name="Test Problem 1",
            dimension=2,
            bounds=np.array([[-2.048, 2.048], [-2.048, 2.048]], dtype=float),
            optimum=np.array([1.0, 1.0], dtype=float),
            optimum_value=0.0,
            function=test_problem_1,
        ),
        "ackley_n2": BenchmarkProblem(
            key="ackley_n2",
            name="Ackley function, n=2",
            dimension=2,
            bounds=np.array([[-32.768, 32.768], [-32.768, 32.768]], dtype=float),
            optimum=np.zeros(2, dtype=float),
            optimum_value=0.0,
            function=ackley,
        ),
        "rastrigin_n5": BenchmarkProblem(
            key="rastrigin_n5",
            name="Rastrigin function, n=5",
            dimension=5,
            bounds=np.array([[-5.12, 5.12]] * 5, dtype=float),
            optimum=np.zeros(5, dtype=float),
            optimum_value=0.0,
            function=rastrigin,
        ),
    }


def problem_table() -> list[dict[str, object]]:
    """Create notebook-friendly metadata rows for the benchmark problems."""
    rows = []
    for problem in get_benchmark_problems().values():
        rows.append(
            {
                "Problem": problem.name,
                "Dimension": problem.dimension,
                "Lower bound": float(problem.bounds[0, 0]),
                "Upper bound": float(problem.bounds[0, 1]),
                "Known optimum f(x*)": problem.optimum_value,
            }
        )
    return rows

