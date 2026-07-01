import torch
import torch.nn as nn


class TemporalAttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1, bias=True)

    def forward(self, H):
        a = torch.softmax(self.score(H).squeeze(-1), dim=1)
        return (H * a.unsqueeze(-1)).sum(dim=1)


class PerVarEncoder(nn.Module):
    def __init__(self, d: int, num_layers: int, dropout: float):
        super().__init__()
        do = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(1, d, batch_first=True, num_layers=num_layers, dropout=do)
        self.pool = TemporalAttnPool(d)

    def forward(self, x):
        H, _ = self.lstm(x)
        return self.pool(H)
