# DP-*f*-FUM: Differentially Private Federated Unlearning

Code and experiments for *"DP-f-FUM: Differentially Private Federated Unlearning via Min-Max Optimization and f-Divergence"*, submitted to ESORICS 2026.

## Problem

Federated Unlearning methods like *f*-FUM can erase a client's contribution from a trained model, but the gradients exchanged during unlearning are vulnerable to reconstruction attacks. DP-*f*-FUM fixes this by adding calibrated Gaussian noise to the unlearning process, providing formal (ε, δ)-differential privacy guarantees via Rényi DP composition.

## Repository

| Notebook | Description |
|----------|-------------|
| `Mnist_FashionMnist_Baseline.ipynb` | MNIST & FashionMNIST unlearning with DP noise sweep (σ ∈ {0, 0.5, 1, 2, 4}), 3 seeds |
| `Mnist_FashionMnist_byzantine.ipynb` | Byzantine robustness on MNIST & FashionMNIST (FedAvg vs median, 20% poisoned) |
| `cifar_final.ipynb` | CIFAR-10 / ResNet-18 unlearning with output perturbation DP |
| `cifar_byzantine.ipynb` | CIFAR-10 Byzantine robustness evaluation |

## Setup

All notebooks run on Google Colab (T4 GPU). Dependencies: PyTorch, NumPy, scikit-learn.

## Results

Strong privacy (ε ≤ 1) costs under 1 percentage point on MNIST and CIFAR-10, and 3.2 points on FashionMNIST. Coordinate-wise median prevents Byzantine collapse where FedAvg drops to random guessing.

## Authors

Izma Khan, Sameed Ahmad, Abdullah Rehman — IBA Karachi
