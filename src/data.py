"""Data loading, synthetic traffic generation, and windowing.

The traffic data is organised as a 3-way tensor  X[day, slot, sensor]:
    day    - calendar day
    slot   - time-of-day interval (e.g. 288 five-minute slots, or 96 / 24 for coarser data)
    sensor - road sensor / location
Flattening (day, slot) gives the usual (T, N) matrix used for sliding windows.
"""
import numpy as np


def make_synthetic_traffic(n_days=60, slots_per_day=96, n_sensors=40, seed=0):
    """Synthetic traffic flow with known low-rank structure + noise.

    Ingredients: daily rush-hour profiles (2 latent patterns), weekday/weekend effect,
    a spatial graph that mixes neighbouring sensors, random incidents and noise.
    Returns X of shape (n_days, slots_per_day, n_sensors).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 24, slots_per_day, endpoint=False)

    def bump(c, w):
        return np.exp(-0.5 * ((t - c) / w) ** 2)

    commute = bump(8, 1.3) + 0.9 * bump(17.5, 1.6)         # rush-hour pattern
    leisure = bump(13, 3.0) + 0.6 * bump(20, 2.0)          # midday / evening pattern
    base = 0.25 + 0.1 * np.sin(2 * np.pi * t / 24)

    # sensors lie on a line/ring; neighbours share behaviour (spatial smoothness)
    pos = np.sort(rng.uniform(0, 1, n_sensors))
    w_commute = 0.5 + 0.5 * np.sin(2 * np.pi * pos * 2)
    w_leisure = 0.5 + 0.5 * np.cos(2 * np.pi * pos * 1.5)
    amp = rng.uniform(0.7, 1.3, n_sensors)

    dow = np.arange(n_days) % 7
    weekend = (dow >= 5).astype(float)
    X = np.zeros((n_days, slots_per_day, n_sensors))
    for d in range(n_days):
        day_scale = 1.0 + 0.08 * rng.standard_normal()
        c_w, l_w = (1 - 0.7 * weekend[d]), (1 + 0.6 * weekend[d])
        prof = (c_w * commute[:, None] * w_commute[None, :]
                + l_w * leisure[:, None] * w_leisure[None, :]
                + base[:, None])
        X[d] = day_scale * prof * amp[None, :]

    # spatial diffusion: each sensor leaks into neighbours, lag of one slot upstream->downstream
    shifted = np.roll(X, 1, axis=1)
    X = 0.8 * X + 0.2 * shifted
    # random incidents: localised drops/spikes that persist several slots
    for _ in range(n_days * 2):
        d, s, n = rng.integers(n_days), rng.integers(slots_per_day - 8), rng.integers(n_sensors)
        X[d, s:s + rng.integers(3, 8), max(0, n - 1):n + 2] *= rng.uniform(0.4, 1.6)
    X += 0.03 * rng.standard_normal(X.shape)
    return np.clip(X, 0, None) * 500.0   # vehicles per interval


def load_npz(path, key="data", slots_per_day=288):
    """Load a real dataset stored as an array (T, N) (e.g. METR-LA / PEMS-BAY exported to .npz).
    Trims to whole days and reshapes into (day, slot, sensor)."""
    arr = np.load(path)[key]
    if arr.ndim == 3:            # (T, N, F) -> keep first feature
        arr = arr[..., 0]
    T = (arr.shape[0] // slots_per_day) * slots_per_day
    return arr[:T].reshape(-1, slots_per_day, arr.shape[1]).astype(np.float64)


def split_days(X, train=0.7, val=0.1):
    D = X.shape[0]
    a, b = int(D * train), int(D * (train + val))
    return X[:a], X[a:b], X[b:]


class Scaler:
    def __init__(self, X):
        self.mean, self.std = X.mean(), X.std() + 1e-8

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse(self, X):
        return X * self.std + self.mean


def to_matrix(X3):
    """(day, slot, sensor) -> (T, sensor)."""
    return X3.reshape(-1, X3.shape[-1])


def make_windows(M, slots_per_day, in_len=12, horizon=12):
    """Sliding windows over a (T, N) matrix.

    Returns
        x     (S, in_len, N)   past observations
        y     (S, horizon, N)  future targets
        t_in  (S, in_len)      time-of-day slot index of each input step
        t_out (S, horizon)     time-of-day slot index of each target step
    """
    T = M.shape[0]
    n = T - in_len - horizon + 1
    idx = np.arange(n)[:, None]
    x = M[idx + np.arange(in_len)[None, :]]
    y = M[idx + in_len + np.arange(horizon)[None, :]]
    slot = np.arange(T) % slots_per_day
    t_in = slot[idx + np.arange(in_len)[None, :]]
    t_out = slot[idx + in_len + np.arange(horizon)[None, :]]
    return x, y, t_in, t_out
