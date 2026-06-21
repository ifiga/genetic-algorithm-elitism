# Genetic Algorithm with Elitism Homework

This project implements a genetic algorithm with elitism from scratch. It includes both binary encoding and real-number encoding, an automatic stagnation-based stopping criterion, 20 independent runs per benchmark and encoding, convergence plots, derivative-method comparisons, and an elitism-vs-no-elitism analysis.

## Run the Notebook

Start Jupyter from the activated environment:

```powershell
jupyter notebook main.ipynb
```

Then run all cells from top to bottom. The notebook imports reusable code from `src/`.

## Outputs

Generated files are saved under `outputs/`:

- `outputs/tables/` contains CSV and LaTeX tables.
- `outputs/figures/` contains PNG convergence and elitism plots.

The main required outputs are:

- `outputs/tables/results_summary.tex`
- `outputs/tables/derivative_comparison.tex`
- `outputs/figures/convergence_test_problem_1.png`
- `outputs/figures/convergence_rastrigin_n5.png`
- `outputs/figures/convergence_ackley_n2.png`
- `outputs/figures/elitism_vs_no_elitism_median.png`
