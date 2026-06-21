"""Derivative-based comparison methods."""

from typing import Iterable, Optional

import numpy as np
from scipy.optimize import minimize

from .benchmark_functions import BenchmarkProblem


def _default_starting_points(problem: BenchmarkProblem) -> list[np.ndarray]:
    bounds = np.asarray(problem.bounds, dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    center = (lower + upper) / 2.0
    points = [
        center,
        lower + 0.25 * (upper - lower),
        lower + 0.75 * (upper - lower),
    ]

    if problem.dimension == 2:
        points.extend(
            [
                np.array([lower[0], upper[1]], dtype=float) * 0.5,
                np.array([upper[0], lower[1]], dtype=float) * 0.5,
                np.array([-1.2, 1.0], dtype=float),
            ]
        )
    else:
        points.append(np.linspace(lower[0] * 0.5, upper[0] * 0.5, problem.dimension))

    return [np.clip(point, lower, upper).astype(float) for point in points]


def run_bfgs_multistart(
    problem: BenchmarkProblem,
    starting_points: Optional[Iterable[np.ndarray]] = None,
    maxiter: int = 2000,
) -> dict[str, object]:
    """Run scipy.optimize.minimize with BFGS from several initial points."""
    starts = list(starting_points) if starting_points is not None else _default_starting_points(problem)
    best_result = None
    best_start = None

    for start in starts:
        result = minimize(
            problem.function,
            np.asarray(start, dtype=float),
            method="BFGS",
            options={"maxiter": maxiter, "gtol": 1e-5},
        )
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result
            best_start = np.asarray(start, dtype=float)

    return {
        "method": "BFGS (multi-start)",
        "best_fitness": float(best_result.fun),
        "best_solution": np.asarray(best_result.x, dtype=float),
        "success": bool(best_result.success),
        "iterations": None if best_result.nit is None else int(best_result.nit),
        "function_evaluations": None if best_result.nfev is None else int(best_result.nfev),
        "starting_point": best_start,
        "message": str(best_result.message),
    }
