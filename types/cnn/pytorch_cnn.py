#!/usr/bin/env python3
"""
PyTorch CNN on a tiny synthetic line-vs-line dataset.

Run: python pytorch_cnn.py
Expected: loss decreases and accuracy rises; CPU is fine.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(n_samples: int = 200, img_size: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.zeros(n_samples, 1, img_size, img_size)
    y = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        if torch.rand(1, generator=g).item() < 0.5:
            col = torch.randint(1, img_size - 1, (1,), generator=g).item()
            X[i, 0, :, col] = 1.0
            y[i] = 0
        else:
            row = torch.randint(1, img_size - 1, (1,), generator=g).item()
            X[i, 0, row, :] = 1.0
            y[i] = 1
    X += 0.05 * torch.randn_like(X, generator=g)
    return X, y


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(4 * 3 * 3, 2),
        )

    def forward(self, x):
        return self.net(x)


def train(epochs: int = 20, lr: float = 0.1, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = make_data()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = SmallCNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss, correct, count = 0.0, 0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            count += yb.size(0)
        print(f"epoch={epoch:02d} loss={total_loss/count:.4f} acc={correct/count:.3f}")


if __name__ == "__main__":
    train()

