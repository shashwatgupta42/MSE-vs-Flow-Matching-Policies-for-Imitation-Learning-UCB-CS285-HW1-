# Imitation Learning on Gym-PushT: MSE vs. Flow Matching Policies (UCB CS285 HW1)

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.9.1+-red.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-PushT--v0-green.svg)](https://gymnasium.farama.org/)
[![WandB](https://img.shields.io/badge/wandb-active-amber.svg)](https://wandb.ai/)

This repository contains an implementation and comparative analysis of two imitation learning methods applied to the **Push-T** robotic manipulation task. The project is from the UCB CS285 (Deep Reinforcement Learning) Homework 1, demonstrating the power of generative policies (Flow Matching) over regression-based behavioral cloning in tasks with multi-modal expert demonstrations.

---

## 1. Overview

The **Push-T** task requires a circular end-effector to push a T-shaped block from a random initial configuration into a fixed target pose. The environment observation space is a 5D state vector (end-effector position + block position & orientation), and the action space is a 2D displacement of the end-effector.

We implement and compare two policy architectures:
1. **Behavioral Cloning (MSE Policy)**: A standard regression policy mapping state inputs directly to action chunks (predicting 8 future actions in one pass) trained via Mean Squared Error (MSE) loss.
2. **Generative Behavioral Cloning (Flow Matching Policy)**: A generative model that learns a conditional velocity field to transport noise to expert actions via Euler ODE integration.

---

## 2. Results & Evaluation

For a comprehensive evaluation, comparative analysis, training curves, rollout videos, and inference step sweeps, please refer to the Jupyter Notebook [report.ipynb](file:///c:/Users/Lenovo/Documents/Projects/UCB_CS285_Deep_RL_HW/HW1/report.ipynb).

---

## 3. Repository Structure

```tree
.
├── data/                       # Downloaded expert trajectory dataset (zarr format)
├── exp/                        # Experiment checkpoints, logs, and generated metrics/plots
├── src/
│   └── hw1_imitation/
│       ├── data.py             # Dataset preprocessing and normalization
│       ├── model.py            # MSE and Flow Matching policy network definitions
│       ├── evaluation.py       # Rollout simulation and performance logging
│       ├── train.py            # Local training entrypoint
│       └── modal_train.py      # Cloud training entrypoint via Modal
├── generate_report_data.py     # Script to generate rollout videos and sweeps
├── plot_logs.py                # Plotting utilities for training curves
├── pyproject.toml              # Project dependencies and environment metadata
└── README.md                   # Project documentation
```

---

## 4. Getting Started

### Prerequisites & Setup
This project uses `uv` for ultra-fast, reproducible package management.

1. **Install `uv`** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Clone the repository** and install dependencies:
   ```bash
   git clone https://github.com/<your-username>/UCB_CS285_Deep_RL_HW1.git
   cd UCB_CS285_Deep_RL_HW1
   uv sync
   ```

Always execute commands prefixed with `uv run` to ensure they run inside the correct virtual environment.

### Weights & Biases (WandB) Login
WandB is used for experiment tracking. Login to your account before launching training:
```bash
uv run wandb login
```

### Training Policies
You can train either policy type locally:

* **Train the MSE Policy**:
  ```bash
  uv run src/hw1_imitation/train.py --policy-type mse --num-epochs 400
  ```
* **Train the Flow Matching Policy**:
  ```bash
  uv run src/hw1_imitation/train.py --policy-type flow --num-epochs 400
  ```

Checkpoints and metrics will be logged inside the `exp/` directory.

### Generating Videos and Sweeps
To evaluate the checkpoints and generate the rollout videos and ODE step sweeps:
```bash
uv run generate_report_data.py
```
This script reads the best checkpoints from `exp/`, performs evaluations, and writes outputs to `exp/best_mse_rollouts.mp4` and `exp/best_flow_rollouts.mp4`.