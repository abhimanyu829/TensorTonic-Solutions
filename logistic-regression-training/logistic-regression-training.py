import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
  """Returns elementwise sigmoid values."""
  return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def train_logistic_regression(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000
) -> tuple[np.ndarray, float]:
  """Returns the trained weights and bias as (w, b)."""
  n_samples, n_features = X.shape

  # 1. Initialize weights as zeros and bias as 0.0
  w = np.zeros(n_features, dtype=float)
  b = 0.0

  # 2. Gradient descent training loop
  for _ in range(steps):
    # Forward pass: compute linear score and predicted probabilities
    z = X @ w + b
    p = _sigmoid(z)

    # Compute gradients of binary cross-entropy loss
    error = p - y
    grad_w = (X.T @ error) / n_samples
    grad_b = float(np.mean(error))

    # Update parameters
    w -= lr * grad_w
    b -= lr * grad_b

  return w, float(b)