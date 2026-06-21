"""Experiment orchestration for the genetic algorithm homework."""

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .benchmark_functions import BenchmarkProblem, get_benchmark_problems
from .binary_ga import bits_per_variable, run_binary_ga
from .derivative_methods import run_bfgs_multistart
from .real_ga import run_real_ga
from .stopping import StagnationConfig


PROBLEM_ORDER = ["test_problem_1", "ackley_n2", "rastrigin_n5"]
PLOT_PROBLEM_ORDER = ["test_problem_1", "rastrigin_n5", "ackley_n2"]
ENCODING_ORDER = ["binary", "real"]

SUMMARY_COLUMNS = [
    ("binary", "test_problem_1", "Binary encoding - Test problem 1"),
    ("real", "test_problem_1", "Real encoding - Test problem 1"),
    ("binary", "ackley_n2", "Binary encoding - Ackley function, n=2"),
    ("real", "ackley_n2", "Real encoding - Ackley function, n=2"),
    ("binary", "rastrigin_n5", "Binary encoding - Rastrigin function, n=5"),
    ("real", "rastrigin_n5", "Real encoding - Rastrigin function, n=5"),
]

CONVERGENCE_FILENAMES = {
    "test_problem_1": "convergence_test_problem_1.png",
    "rastrigin_n5": "convergence_rastrigin_n5.png",
    "ackley_n2": "convergence_ackley_n2.png",
}

DEFAULT_POPULATION_SIZE = 100
DEFAULT_PC = 0.9
DEFAULT_ELITE_SIZE = 1
DEFAULT_ETA_C = 20.0
DEFAULT_ETA_M = 20.0


def ensure_output_dirs(output_dir: str | Path = "outputs") -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _format_solution(solution: Iterable[float], digits: int = 8) -> str:
    return "[" + ", ".join(f"{float(value):.{digits}g}" for value in solution) + "]"


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _format_cell(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return _latex_escape(value)


def dataframe_to_latex(df: pd.DataFrame) -> str:
    """Create a small LaTeX tabular without requiring optional pandas dependencies."""
    column_spec = "l" * len(df.columns)
    lines = [rf"\begin{{tabular}}{{{column_spec}}}", r"\hline"]
    lines.append(" & ".join(_latex_escape(col) for col in df.columns) + r" \\")
    lines.append(r"\hline")
    for _, row in df.iterrows():
        lines.append(" & ".join(_format_cell(value) for value in row.tolist()) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, csv_path: str | Path, tex_path: str | Path) -> None:
    csv_path = Path(csv_path)
    tex_path = Path(tex_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    tex_path.write_text(dataframe_to_latex(df), encoding="utf-8")


def seed_for(problem_key: str, encoding: str, run_index: int, phase: str = "main") -> int:
    phase_base = {"pilot": 1000, "main": 10000, "elitism": 30000}.get(phase, 50000)
    problem_offset = PROBLEM_ORDER.index(problem_key) * 5000
    encoding_offset = 0 if encoding == "binary" else 1000
    return int(phase_base + problem_offset + encoding_offset + run_index)


def run_ga_case(
    problem: BenchmarkProblem,
    encoding: str,
    seed: int,
    max_generations: int,
    elite_size: int = DEFAULT_ELITE_SIZE,
    stopping_config: Optional[StagnationConfig] = None,
) -> dict[str, object]:
    if encoding == "binary":
        return run_binary_ga(
            problem.function,
            problem.bounds,
            population_size=DEFAULT_POPULATION_SIZE,
            pc=DEFAULT_PC,
            elite_size=elite_size,
            max_generations=max_generations,
            stopping_config=stopping_config,
            seed=seed,
        )
    if encoding == "real":
        return run_real_ga(
            problem.function,
            problem.bounds,
            population_size=DEFAULT_POPULATION_SIZE,
            pc=DEFAULT_PC,
            elite_size=elite_size,
            max_generations=max_generations,
            stopping_config=stopping_config,
            seed=seed,
            eta_c=DEFAULT_ETA_C,
            eta_m=DEFAULT_ETA_M,
        )
    raise ValueError(f"Unknown encoding: {encoding}")


def _result_row(
    problem: BenchmarkProblem,
    encoding: str,
    run_index: int,
    seed: int,
    result: dict[str, object],
    phase: str,
) -> dict[str, object]:
    metadata = result["metadata"]
    return {
        "phase": phase,
        "problem_key": problem.key,
        "problem": problem.name,
        "encoding": encoding,
        "run": int(run_index),
        "seed": int(seed),
        "final_best_fitness": float(result["best_fitness"]),
        "best_solution": _format_solution(result["best_solution"]),
        "stopped_generation": int(result["stopped_generation"]),
        "function_evaluations": int(metadata["function_evaluations"]),
        "elite_size": int(metadata["elite_size"]),
        "mutation_probability": float(metadata["pm"]),
    }


def run_pilot_experiments(
    pilot_runs_per_case: int = 3,
    stopping_config: Optional[StagnationConfig] = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], np.ndarray]]:
    stopping_config = stopping_config or StagnationConfig()
    problems = get_benchmark_problems()
    rows = []
    histories: dict[tuple[str, str, int], np.ndarray] = {}

    for problem_key in PROBLEM_ORDER:
        problem = problems[problem_key]
        for encoding in ENCODING_ORDER:
            for run_index in range(1, pilot_runs_per_case + 1):
                seed = seed_for(problem_key, encoding, run_index, phase="pilot")
                result = run_ga_case(
                    problem,
                    encoding,
                    seed=seed,
                    max_generations=stopping_config.max_generations,
                    elite_size=DEFAULT_ELITE_SIZE,
                    stopping_config=stopping_config,
                )
                rows.append(_result_row(problem, encoding, run_index, seed, result, phase="pilot"))
                histories[(problem_key, encoding, run_index)] = result["best_fitness_history"]

    return pd.DataFrame(rows), histories


def select_generation_counts(
    pilot_results: pd.DataFrame,
    stopping_config: Optional[StagnationConfig] = None,
    round_to: int = 25,
    buffer_generations: int = 25,
) -> tuple[dict[str, int], pd.DataFrame]:
    stopping_config = stopping_config or StagnationConfig()
    problems = get_benchmark_problems()
    generation_counts: dict[str, int] = {}
    rows = []

    for problem_key in PROBLEM_ORDER:
        subset = pilot_results[pilot_results["problem_key"] == problem_key]
        max_stopped = int(subset["stopped_generation"].max())
        selected = int(np.ceil((max_stopped + buffer_generations) / round_to) * round_to)
        selected = max(stopping_config.min_generations, min(stopping_config.max_generations, selected))
        generation_counts[problem_key] = selected
        rows.append(
            {
                "Problem": problems[problem_key].name,
                "Max pilot stopped generation": max_stopped,
                "Selected fixed generations": selected,
                "Selection rule": f"ceil((max pilot stop + {buffer_generations}) / {round_to}) * {round_to}",
            }
        )

    return generation_counts, pd.DataFrame(rows)


def run_independent_experiments(
    generation_counts: Dict[str, int],
    runs_per_case: int = 20,
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], np.ndarray]]:
    problems = get_benchmark_problems()
    rows = []
    histories: dict[tuple[str, str, int], np.ndarray] = {}

    for problem_key in PROBLEM_ORDER:
        problem = problems[problem_key]
        max_generations = int(generation_counts[problem_key])
        for encoding in ENCODING_ORDER:
            for run_index in range(1, runs_per_case + 1):
                seed = seed_for(problem_key, encoding, run_index, phase="main")
                result = run_ga_case(
                    problem,
                    encoding,
                    seed=seed,
                    max_generations=max_generations,
                    elite_size=DEFAULT_ELITE_SIZE,
                    stopping_config=None,
                )
                rows.append(_result_row(problem, encoding, run_index, seed, result, phase="main"))
                histories[(problem_key, encoding, run_index)] = result["best_fitness_history"]

    return pd.DataFrame(rows), histories


def build_results_summary(results: pd.DataFrame, runs_per_case: int = 20) -> pd.DataFrame:
    summary = pd.DataFrame({"Experiment": [f"Experiment {i}" for i in range(1, runs_per_case + 1)]})

    for encoding, problem_key, column_name in SUMMARY_COLUMNS:
        subset = results[(results["problem_key"] == problem_key) & (results["encoding"] == encoding)]
        subset = subset.sort_values("run")
        values = subset["final_best_fitness"].to_numpy(dtype=float)
        if len(values) != runs_per_case:
            raise ValueError(f"Expected {runs_per_case} runs for {encoding} {problem_key}, got {len(values)}")
        summary[column_name] = values

    statistic_rows = []
    numeric_columns = [column for column in summary.columns if column != "Experiment"]
    for label, function in [
        ("Mean", np.mean),
        ("Standard Deviation", lambda values: np.std(values, ddof=1)),
        ("Min", np.min),
        ("Max", np.max),
    ]:
        row = {"Experiment": label}
        for column in numeric_columns:
            row[column] = float(function(summary[column].to_numpy(dtype=float)))
        statistic_rows.append(row)

    return pd.concat([summary, pd.DataFrame(statistic_rows)], ignore_index=True)


def select_median_representative_runs(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for problem_key in PLOT_PROBLEM_ORDER:
        for encoding in ENCODING_ORDER:
            subset = results[(results["problem_key"] == problem_key) & (results["encoding"] == encoding)].copy()
            median_value = float(subset["final_best_fitness"].median())
            subset["distance_to_median"] = np.abs(subset["final_best_fitness"] - median_value)
            chosen = subset.sort_values(["distance_to_median", "run"]).iloc[0]
            rows.append(
                {
                    "Problem": chosen["problem"],
                    "problem_key": problem_key,
                    "encoding": encoding,
                    "Selected run": int(chosen["run"]),
                    "Final best fitness": float(chosen["final_best_fitness"]),
                    "Selection rule": "Run closest to median final best fitness within each problem and encoding",
                }
            )
    return pd.DataFrame(rows)


def build_derivative_comparison(results: pd.DataFrame) -> pd.DataFrame:
    problems = get_benchmark_problems()
    rows = []
    for problem_key in ["test_problem_1", "ackley_n2"]:
        problem = problems[problem_key]
        for encoding in ENCODING_ORDER:
            subset = results[(results["problem_key"] == problem_key) & (results["encoding"] == encoding)]
            best = subset.sort_values("final_best_fitness").iloc[0]
            method_encoding = "Binary" if encoding == "binary" else "Real"
            rows.append(
                {
                    "Problem": problem.name,
                    "Method": f"GA {method_encoding} with elitism",
                    "Best fitness": float(best["final_best_fitness"]),
                    "Best solution": best["best_solution"],
                    "Success": "Not applicable",
                    "Iterations / evaluations": f"{int(best['function_evaluations'])} evaluations",
                }
            )

        bfgs_result = run_bfgs_multistart(problem)
        rows.append(
            {
                "Problem": problem.name,
                "Method": "BFGS (multi-start)",
                "Best fitness": float(bfgs_result["best_fitness"]),
                "Best solution": _format_solution(bfgs_result["best_solution"]),
                "Success": bool(bfgs_result["success"]),
                "Iterations / evaluations": (
                    f"{bfgs_result['iterations']} iterations / "
                    f"{bfgs_result['function_evaluations']} evaluations"
                ),
            }
        )

    return pd.DataFrame(rows)


def run_elitism_benefit_experiment(
    generation_count: int,
    runs: int = 20,
    problem_key: str = "rastrigin_n5",
    encoding: str = "real",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    problems = get_benchmark_problems()
    problem = problems[problem_key]
    rows = []
    histories: dict[tuple[str, int], np.ndarray] = {}

    for label, elite_size in [("GA with elitism", 1), ("GA without elitism", 0)]:
        for run_index in range(1, runs + 1):
            seed = seed_for(problem_key, encoding, run_index, phase="elitism")
            result = run_ga_case(
                problem,
                encoding,
                seed=seed,
                max_generations=int(generation_count),
                elite_size=elite_size,
                stopping_config=None,
            )
            rows.append(_result_row(problem, encoding, run_index, seed, result, phase=label))
            histories[(label, run_index)] = result["best_fitness_history"]

    median_rows = {"Generation": np.arange(int(generation_count) + 1)}
    for label in ["GA with elitism", "GA without elitism"]:
        matrix = np.vstack([histories[(label, run_index)] for run_index in range(1, runs + 1)])
        median_rows[label] = np.median(matrix, axis=0)

    return pd.DataFrame(rows), pd.DataFrame(median_rows), histories


def binary_encoding_table() -> pd.DataFrame:
    problems = get_benchmark_problems()
    rows = []
    for problem_key in PROBLEM_ORDER:
        problem = problems[problem_key]
        bits = bits_per_variable(problem.bounds)
        rows.append(
            {
                "Problem": problem.name,
                "Bits per variable": bits.tolist(),
                "Chromosome length": int(np.sum(bits)),
                "Binary mutation probability": 1.0 / float(np.sum(bits)),
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(output_dir: str | Path = "outputs") -> pd.DataFrame:
    output_dir = Path(output_dir)
    required_files = [
        output_dir / "tables" / "results_summary.csv",
        output_dir / "tables" / "results_summary.tex",
        output_dir / "tables" / "derivative_comparison.csv",
        output_dir / "tables" / "derivative_comparison.tex",
        output_dir / "figures" / "convergence_test_problem_1.png",
        output_dir / "figures" / "convergence_rastrigin_n5.png",
        output_dir / "figures" / "convergence_ackley_n2.png",
        output_dir / "figures" / "elitism_vs_no_elitism_median.png",
    ]
    rows = []
    for path in required_files:
        rows.append({"Check": str(path).replace("\\", "/"), "Status": "OK" if path.exists() else "Missing"})

    summary_path = output_dir / "tables" / "results_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        expected_rows = 24
        rows.append(
            {
                "Check": "results_summary.csv has 20 experiments plus 4 statistic rows",
                "Status": "OK" if len(summary) == expected_rows else f"Expected {expected_rows}, got {len(summary)}",
            }
        )

    derivative_path = output_dir / "tables" / "derivative_comparison.csv"
    if derivative_path.exists():
        derivative = pd.read_csv(derivative_path)
        required_problems = {"Test Problem 1", "Ackley function, n=2"}
        rows.append(
            {
                "Check": "derivative comparison includes Test Problem 1 and Ackley n=2",
                "Status": "OK" if required_problems.issubset(set(derivative["Problem"])) else "Missing problem",
            }
        )

    return pd.DataFrame(rows)
