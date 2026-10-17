# Error-Aware Quantum Risk Engine (Nifty, IQAE)

## Project Overview

This project investigates the use of Iterative Quantum Amplitude Estimation (IQAE) for European call option pricing using a discretized Nifty 50 terminal-price distribution.

The project compares quantum amplitude estimation with classical Monte Carlo simulation and studies the effects of discretization, state preparation, circuit resources, and noise on estimation accuracy.

The implementation will initially focus on small-scale quantum circuit simulations using Qiskit and Qiskit Aer on a CPU.

## Objectives

- Construct a risk-neutral probability distribution for Nifty 50 terminal prices.
- Discretize the distribution into a finite probability grid.
- Implement quantum state preparation and a payoff oracle.
- Estimate expected option payoffs using IQAE.
- Compare IQAE results with classical Monte Carlo estimates.
- Analyse discretization error, state-preparation costs, circuit depth, and noise sensitivity.
- Investigate matrix product state (MPS) loading if time permits.

## Team Responsibilities

### Person 1 — Market Data and State Preparation
- Market data collection and preprocessing
- Risk-neutral probability distributions
- Probability-grid construction and discretization analysis
- Exact quantum state loader
- Optional MPS loader

### Person 2 — Quantum Algorithm
- Payoff oracle implementation
- IQAE implementation
- Quantum circuit validation
- Integration with Person 1's state preparation

### Person 3 — Classical Baseline and Analysis
- Classical Monte Carlo pricing
- Error analysis and crossover experiments
- Circuit resource and performance comparisons
- Results visualisation and reporting

## Technology Stack

- Python
- Jupyter Notebook
- VS Code
- Qiskit
- Qiskit Aer
- NumPy and SciPy
- Matplotlib
- Git and GitHub

## Repository Structure

```text
error-aware-quantum-risk-engine/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
├── person1_market_data/
├── person2_quantum_engine/
├── person3_analysis/
├── notebooks/
├── tests/
├── results/
└── docs/
```

## Getting Started

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the project's pinned dependencies.
4. Select the environment as the Jupyter kernel in VS Code.

Detailed installation instructions will be added after the team validates the development environment.

## Reproducibility

Experiments should record their parameters, random seeds where applicable, package versions, and output metrics.

Do not commit virtual environments, credentials, or large raw market-data files.

## Project Status

In development. Results, benchmarks, and conclusions will be added after implementation and validation.

## Disclaimer

This is an academic quantum algorithm simulation project. Simulated results do not establish practical quantum advantage or constitute financial advice.
