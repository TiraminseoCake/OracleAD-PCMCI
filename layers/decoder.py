import torch
import torch.nn as nn


class PerVarReconDecoder(nn.Module):
    def __init__(self, d: int, L: int, num_layers: int, dropout: float):
        super().__init__()
        self.out_len = L - 1
        self.d = d
        self.num_layers = num_layers
        do = dropout if num_layers > 1 else 0.0

        self.init_h = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, num_layers * d),
        )
        self.init_c = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, num_layers * d),
        )

        self.lstm = nn.LSTM(1, d, batch_first=True, num_layers=num_layers, dropout=do)
        self.out = nn.Linear(d, 1)

    def forward(self, c):
        B, d = c.shape
        z = torch.zeros(B, self.out_len, 1, device=c.device, dtype=c.dtype)
        h0 = torch.tanh(self.init_h(c)).view(self.num_layers, B, d).contiguous()
        c0 = torch.tanh(self.init_c(c)).view(self.num_layers, B, d).contiguous()
        Y, _ = self.lstm(z, (h0, c0))
        O = self.out(Y).squeeze(-1)
        return O
