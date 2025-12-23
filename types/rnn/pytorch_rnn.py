#!/usr/bin/env python3
"""
PyTorch RNN for sequence parity classification (even/odd count of 1s).

Run: python pytorch_rnn.py
Expected: loss drops and accuracy rises; CPU works fine.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(n_samples: int = 500, seq_len: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 2, (n_samples, seq_len), generator=g)
    y = (X.sum(dim=1) % 2).long()
    # one-hot encode tokens for RNN input
    X_one_hot = torch.zeros(n_samples, seq_len, 2)
    X_one_hot.scatter_(2, X.unsqueeze(-1), 1.0)
    return X_one_hot, y


class ParityRNN(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.rnn = nn.RNN(input_size=2, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)


def train(epochs: int = 25, lr: float = 0.05, batch_size: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = make_data()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = ParityRNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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

