#!/usr/bin/env python3
"""
Vanilla RNN from scratch in NumPy for sequence parity classification.

Task: read a binary sequence and predict whether the number of 1s is even/odd.
Run: python numpy_rnn.py
Expected: loss falls and accuracy rises over epochs.
"""

import numpy as np


def make_data(n_samples: int = 200, seq_len: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_samples, seq_len))
    y = (X.sum(axis=1) % 2).astype(np.int64)  # 0 -> even, 1 -> odd
    return X, y


def one_hot(idx: int, depth: int = 2):
    v = np.zeros((1, depth), dtype=np.float32)
    v[0, idx] = 1.0
    return v


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def init_params(input_dim=2, hidden=16, num_classes=2, seed: int = 0):
    rng = np.random.default_rng(seed)
    params = {
        "Wxh": 0.2 * rng.standard_normal((input_dim, hidden)).astype(np.float32),
        "Whh": 0.2 * rng.standard_normal((hidden, hidden)).astype(np.float32),
        "bh": np.zeros((1, hidden), dtype=np.float32),
        "Why": 0.2 * rng.standard_normal((hidden, num_classes)).astype(np.float32),
        "by": np.zeros((1, num_classes), dtype=np.float32),
    }
    return params


def forward(seq, params):
    h_prev = np.zeros((1, params["Whh"].shape[0]), dtype=np.float32)
    xs, hs = [], [h_prev]
    for token in seq:
        x_t = one_hot(int(token))
        h_prev = np.tanh(x_t @ params["Wxh"] + h_prev @ params["Whh"] + params["bh"])
        xs.append(x_t)
        hs.append(h_prev)
    scores = h_prev @ params["Why"] + params["by"]
    probs = softmax(scores)
    cache = {"xs": xs, "hs": hs, "probs": probs}
    return probs, cache


def compute_loss(probs, target):
    loss = -np.log(probs[0, target] + 1e-9)
    return float(loss)


def backward(params, cache, target):
    xs, hs, probs = cache["xs"], cache["hs"], cache["probs"]
    T = len(xs)
    grads = {k: np.zeros_like(v) for k, v in params.items()}

    dscores = probs.copy()
    dscores[0, target] -= 1  # derivative of cross-entropy

    grads["Why"] += hs[-1].T @ dscores
    grads["by"] += dscores
    dh_next = dscores @ params["Why"].T

    for t in reversed(range(T)):
        h_t = hs[t + 1]
        h_prev = hs[t]
        dh = dh_next
        dh_raw = dh * (1 - h_t**2)  # tanh derivative

        grads["bh"] += dh_raw
        grads["Wxh"] += xs[t].T @ dh_raw
        grads["Whh"] += h_prev.T @ dh_raw
        dh_next = dh_raw @ params["Whh"].T
    return grads


def update(params, grads, lr=0.05):
    for k in params:
        params[k] -= lr * grads[k]


def accuracy(params, X, y):
    correct = 0
    for seq, label in zip(X, y):
        probs, _ = forward(seq, params)
        pred = int(probs.argmax(axis=1))
        correct += int(pred == label)
    return correct / len(y)


def train(epochs: int = 40, lr: float = 0.05):
    X, y = make_data()
    params = init_params()
    for epoch in range(epochs):
        total_loss = 0.0
        # stochastic training over individual sequences
        for seq, target in zip(X, y):
            probs, cache = forward(seq, params)
            loss = compute_loss(probs, int(target))
            grads = backward(params, cache, int(target))
            update(params, grads, lr)
            total_loss += loss
        if epoch % 5 == 0 or epoch == epochs - 1:
            acc = accuracy(params, X, y)
            print(f"epoch={epoch:02d} loss={total_loss/len(X):.4f} acc={acc:.3f}")
    return params


if __name__ == "__main__":
    train()

