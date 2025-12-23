#!/usr/bin/env python3
"""
Minimal 2-layer MLP built only with NumPy on a toy XOR-like dataset.

Run: python numpy_mlp.py
Expected: loss decreases and accuracy climbs toward ~95%+.
"""

import numpy as np


def make_data(n_samples: int = 400, seed: int = 0):
    """Generate 2D points in four quadrants with XOR-style labels."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)  # label 1 if both coords share sign
    return X, y


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def init_params(input_dim=2, hidden=16, num_classes=2, seed: int = 42):
    rng = np.random.default_rng(seed)
    params = {
        "W1": 0.1 * rng.standard_normal((input_dim, hidden)),
        "b1": np.zeros((1, hidden)),
        "W2": 0.1 * rng.standard_normal((hidden, num_classes)),
        "b2": np.zeros((1, num_classes)),
    }
    return params


def forward(X: np.ndarray, params: dict):
    """Forward pass through affine -> tanh -> affine -> softmax."""
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.tanh(z1)
    scores = a1 @ params["W2"] + params["b2"]
    probs = softmax(scores)
    cache = {"X": X, "z1": z1, "a1": a1, "probs": probs}
    return probs, cache


def compute_loss(probs: np.ndarray, y: np.ndarray) -> float:
    N = y.shape[0]
    correct = -np.log(probs[np.arange(N), y] + 1e-9)
    return float(correct.mean())


def backward(params: dict, cache: dict, y: np.ndarray):
    """Backprop for the 2-layer network."""
    X, a1, probs = cache["X"], cache["a1"], cache["probs"]
    N = X.shape[0]

    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N

    grads = {}
    grads["W2"] = a1.T @ dscores
    grads["b2"] = dscores.sum(axis=0, keepdims=True)

    da1 = dscores @ params["W2"].T
    dz1 = da1 * (1 - np.tanh(cache["z1"]) ** 2)
    grads["W1"] = X.T @ dz1
    grads["b1"] = dz1.sum(axis=0, keepdims=True)
    return grads


def update(params: dict, grads: dict, lr: float = 0.1):
    for k in params:
        params[k] -= lr * grads[k]


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    preds = probs.argmax(axis=1)
    return float((preds == y).mean())


def train(epochs: int = 400, lr: float = 0.1, hidden: int = 16):
    X, y = make_data()
    params = init_params(hidden=hidden)
    for step in range(epochs):
        probs, cache = forward(X, params)
        loss = compute_loss(probs, y)
        grads = backward(params, cache, y)
        update(params, grads, lr)
        if step % 50 == 0 or step == epochs - 1:
            acc = accuracy(probs, y)
            print(f"step={step:03d} loss={loss:.4f} acc={acc:.3f}")
    return params


if __name__ == "__main__":
    train()

