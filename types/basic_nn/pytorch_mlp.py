#!/usr/bin/env python3
"""
PyTorch version of a small MLP on an XOR-like dataset.

Run: python pytorch_mlp.py
Expected: loss drops and accuracy heads toward ~95%+ on the toy data.
Dependencies: torch (CPU is fine).
"""

import torch
from torch import nn


def make_data(n_samples: int = 400, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, 2, generator=g)
    y = ((X[:, 0] * X[:, 1]) > 0).long()
    return X, y


class MLP(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def train(epochs: int = 300, lr: float = 0.1, hidden: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = make_data()
    X, y = X.to(device), y.to(device)

    model = MLP(hidden).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for step in range(epochs):
        opt.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == epochs - 1:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == y).float().mean().item()
            print(f"step={step:03d} loss={loss.item():.4f} acc={acc:.3f}")


if __name__ == "__main__":
    train()

