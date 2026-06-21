"""Plotting utilities for the homework notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_convergence(
    problem_name: str,
    binary_history: np.ndarray,
    real_history: np.ndarray,
    output_path: str | Path,
):
    """Plot one representative convergence curve for each encoding."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(len(binary_history)), binary_history, label="Binary encoding", linewidth=2.0)
    ax.plot(np.arange(len(real_history)), real_history, label="Real encoding", linewidth=2.0)
    ax.set_title(problem_name)
    ax.set_xlabel("Number of generations")
    ax.set_ylabel("Best individual fitness value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    return fig, ax


def plot_elitism_median(median_history: pd.DataFrame, output_path: str | Path):
    """Plot median convergence with and without elitism."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        median_history["Generation"],
        median_history["GA with elitism"],
        label="GA with elitism",
        linewidth=2.0,
    )
    ax.plot(
        median_history["Generation"],
        median_history["GA without elitism"],
        label="GA without elitism",
        linewidth=2.0,
    )
    ax.set_title("Rastrigin n=5: median convergence with and without elitism")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Median best individual fitness value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    return fig, ax

