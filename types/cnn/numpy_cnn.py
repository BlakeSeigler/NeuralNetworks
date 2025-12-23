#!/usr/bin/env python3
"""
Tiny CNN from scratch in NumPy (conv + ReLU + maxpool + linear).

Run: python numpy_cnn.py
Expected: loss decreases and accuracy rises on a toy line-vs-line dataset.
"""

import numpy as np


def make_data(n_samples: int = 80, img_size: int = 8, seed: int = 0):
    """
    Build binary images of size img_size x img_size with either a vertical
    or horizontal bright line. Label 0 -> vertical, 1 -> horizontal.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, 1, img_size, img_size), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        if rng.random() < 0.5:
            col = rng.integers(1, img_size - 1)
            X[i, 0, :, col] = 1.0
            y[i] = 0
        else:
            row = rng.integers(1, img_size - 1)
            X[i, 0, row, :] = 1.0
            y[i] = 1
    X += 0.05 * rng.standard_normal(X.shape)  # small noise
    return X, y


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def conv_forward(X, W, b):
    N, C, H, W_in = X.shape
    F, _, KH, KW = W.shape
    out_h = H - KH + 1
    out_w = W_in - KW + 1
    out = np.zeros((N, F, out_h, out_w), dtype=np.float32)
    for n in range(N):
        for f in range(F):
            for i in range(out_h):
                for j in range(out_w):
                    region = X[n, :, i : i + KH, j : j + KW]
                    out[n, f, i, j] = np.sum(region * W[f]) + b[f]
    cache = (X, W, b)
    return out, cache


def conv_backward(dout, cache):
    X, W, b = cache
    N, C, H, W_in = X.shape
    F, _, KH, KW = W.shape
    _, _, out_h, out_w = dout.shape

    dX = np.zeros_like(X)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            for i in range(out_h):
                for j in range(out_w):
                    region = X[n, :, i : i + KH, j : j + KW]
                    dW[f] += dout[n, f, i, j] * region
                    dX[n, :, i : i + KH, j : j + KW] += dout[n, f, i, j] * W[f]
                    db[f] += dout[n, f, i, j]
    return dX, dW, db


def relu_forward(X):
    out = np.maximum(0, X)
    cache = X
    return out, cache


def relu_backward(dout, cache):
    X = cache
    dX = dout * (X > 0)
    return dX


def maxpool_forward(X, pool=2):
    N, C, H, W = X.shape
    out_h, out_w = H // pool, W // pool
    out = np.zeros((N, C, out_h, out_w), dtype=np.float32)
    mask = np.zeros_like(X, dtype=bool)
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    hs, ws = i * pool, j * pool
                    window = X[n, c, hs : hs + pool, ws : ws + pool]
                    idx = np.unravel_index(np.argmax(window), window.shape)
                    out[n, c, i, j] = window[idx]
                    mask[n, c, hs + idx[0], ws + idx[1]] = True
    cache = (mask, pool)
    return out, cache


def maxpool_backward(dout, cache):
    mask, pool = cache
    N, C, out_h, out_w = dout.shape
    dX = np.zeros_like(mask, dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    hs, ws = i * pool, j * pool
                    # Only the max location receives gradient
                    dX[n, c, hs : hs + pool, ws : ws + pool] += dout[n, c, i, j] * mask[
                        n, c, hs : hs + pool, ws : ws + pool
                    ]
    return dX


def linear_forward(X, W, b):
    out = X @ W + b
    cache = (X, W, b)
    return out, cache


def linear_backward(dout, cache):
    X, W, _ = cache
    dX = dout @ W.T
    dW = X.T @ dout
    db = dout.sum(axis=0, keepdims=True)
    return dX, dW, db


def compute_loss(probs, y):
    N = y.shape[0]
    loss = -np.log(probs[np.arange(N), y] + 1e-9).mean()
    return float(loss)


def accuracy(probs, y):
    preds = probs.argmax(axis=1)
    return float((preds == y).mean())


def init_params(in_channels=1, num_filters=4, kernel_size=3, pooled_side=3, num_classes=2, seed=1):
    rng = np.random.default_rng(seed)
    W_conv = 0.1 * rng.standard_normal((num_filters, in_channels, kernel_size, kernel_size)).astype(np.float32)
    b_conv = np.zeros((num_filters,), dtype=np.float32)
    flat_dim = num_filters * pooled_side * pooled_side
    W_fc = 0.1 * rng.standard_normal((flat_dim, num_classes)).astype(np.float32)
    b_fc = np.zeros((1, num_classes), dtype=np.float32)
    return {"Wc": W_conv, "bc": b_conv, "Wf": W_fc, "bf": b_fc}


def forward_pass(X, params):
    conv_out, conv_cache = conv_forward(X, params["Wc"], params["bc"])
    relu_out, relu_cache = relu_forward(conv_out)
    pool_out, pool_cache = maxpool_forward(relu_out, pool=2)
    flat = pool_out.reshape(X.shape[0], -1)
    scores, lin_cache = linear_forward(flat, params["Wf"], params["bf"])
    probs = softmax(scores)
    cache = {
        "conv": conv_cache,
        "relu": relu_cache,
        "pool": pool_cache,
        "flat": flat,
        "lin": lin_cache,
        "probs": probs,
    }
    return probs, cache


def backward_pass(params, cache, y):
    N = y.shape[0]
    probs = cache["probs"]
    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N

    dflat, dWf, dbf = linear_backward(dscores, cache["lin"])
    dpool = dflat.reshape(cache["pool"][0].shape)
    drelu = maxpool_backward(dpool, cache["pool"])
    dconv = relu_backward(drelu, cache["relu"])
    dX, dWc, dbc = conv_backward(dconv, cache["conv"])
    grads = {"Wc": dWc, "bc": dbc, "Wf": dWf, "bf": dbf}
    return grads, dX


def update(params, grads, lr=0.1):
    for k in params:
        params[k] -= lr * grads[k]


def train(epochs: int = 120, lr: float = 0.2, batch_size: int = 16):
    X, y = make_data()
    params = init_params()
    N = X.shape[0]
    for step in range(epochs):
        # simple minibatch SGD
        idx = np.random.choice(N, batch_size, replace=False)
        Xb, yb = X[idx], y[idx]
        probs, cache = forward_pass(Xb, params)
        loss = compute_loss(probs, yb)
        grads, _ = backward_pass(params, cache, yb)
        update(params, grads, lr)

        if step % 20 == 0 or step == epochs - 1:
            full_probs, _ = forward_pass(X, params)
            acc = accuracy(full_probs, y)
            print(f"step={step:03d} loss={loss:.4f} acc={acc:.3f}")
    return params


if __name__ == "__main__":
    train()

