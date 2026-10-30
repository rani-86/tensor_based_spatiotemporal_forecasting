"""Forecasting models. All neural models map (B, in_len, ...) -> (B, horizon, N)."""
import torch
import torch.nn as nn


class FullGRU(nn.Module):
    """Conventional baseline: GRU on the raw N-dimensional sensor vector."""

    def __init__(self, n_sensors, hidden=128, horizon=12, layers=2):
        super().__init__()
        self.horizon, self.n = horizon, n_sensors
        self.rnn = nn.GRU(n_sensors, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, horizon * n_sensors)

    def forward(self, x, *_):
        h, _ = self.rnn(x)
        return self.head(h[:, -1]).view(-1, self.horizon, self.n)


class FullLSTM(FullGRU):
    def __init__(self, n_sensors, hidden=128, horizon=12, layers=2):
        super().__init__(n_sensors, hidden, horizon, layers)
        self.rnn = nn.LSTM(n_sensors, hidden, layers, batch_first=True)


class TensorLatentNet(nn.Module):
    """Forecast in the low-rank latent space of the tensor decomposition.

    input:  z_in  (B, in_len, r_n) latent coefficients (projection of sensor vectors)
            e_in  (B, in_len, r_s) time-of-day embedding of input steps (temporal factor)
            e_out (B, horizon, r_s) time-of-day embedding of target steps (known in advance)
    output: z_hat (B, horizon, r_n); lifted to sensor space with the fixed spatial factor.
    `cell` selects the temporal network: 'gru' | 'lstm' | 'tcn'.
    """

    def __init__(self, r_n, r_s, hidden=64, horizon=12, cell="gru", layers=2, in_len=12):
        super().__init__()
        self.horizon, self.r_n, self.cell = horizon, r_n, cell
        d_in = r_n + r_s
        if cell in ("gru", "lstm"):
            RNN = nn.GRU if cell == "gru" else nn.LSTM
            self.enc = RNN(d_in, hidden, layers, batch_first=True)
            enc_out = hidden
        elif cell == "tcn":
            chans, blocks, c = hidden, [], d_in
            for i in range(layers):
                blocks += [nn.Conv1d(c, chans, 3, padding=2 ** i, dilation=2 ** i), nn.ReLU()]
                c = chans
            self.enc = nn.Sequential(*blocks)
            enc_out = chans
        else:
            raise ValueError(cell)
        self.head = nn.Sequential(nn.Linear(enc_out + horizon * r_s, hidden), nn.ReLU(),
                                  nn.Linear(hidden, horizon * r_n))

    def forward(self, z_in, e_in, e_out):
        u = torch.cat([z_in, e_in], dim=-1)
        if self.cell == "tcn":
            h = self.enc(u.transpose(1, 2))[:, :, -1]
        else:
            h = self.enc(u)[0][:, -1]
        h = torch.cat([h, e_out.flatten(1)], dim=-1)
        return self.head(h).view(-1, self.horizon, self.r_n)
