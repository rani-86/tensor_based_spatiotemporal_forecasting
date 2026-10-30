"""CP / Tucker decomposition of the traffic tensor and the projection operators built on it."""
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker

tl.set_backend("numpy")


class TensorBasis:
    """Low-rank spatial/temporal basis learned from the TRAINING tensor only (no leakage).

    X[day, slot, sensor] ~= sum  core * A_day (x) A_slot (x) A_sensor
    * A_sensor  (N x r_n): spatial basis  -> project a sensor vector x_t in R^N to z_t in R^{r_n}
    * A_slot    (S x r_s): time-of-day embedding -> used as extra input features
    """

    def __init__(self, method="tucker", rank=(8, 8, 8), n_iter=200, seed=0):
        assert method in ("tucker", "cp")
        self.method, self.rank, self.n_iter, self.seed = method, rank, n_iter, seed

    def fit(self, X3):
        if self.method == "tucker":
            r = self.rank if isinstance(self.rank, (tuple, list)) else (self.rank,) * 3
            r = tuple(min(a, b) for a, b in zip(r, X3.shape))
            core, factors = tucker(X3, rank=r, n_iter_max=self.n_iter, init="svd",
                                   random_state=self.seed, tol=1e-6)
            self.core, self.factors = core, factors
            self.recon_ = tl.tucker_to_tensor((core, factors))
        else:
            R = self.rank if isinstance(self.rank, int) else self.rank[0]
            cp = parafac(X3, rank=R, n_iter_max=self.n_iter, init="svd",
                         random_state=self.seed, tol=1e-7, normalize_factors=True)
            self.weights, self.factors = cp.weights, cp.factors
            self.recon_ = tl.cp_to_tensor(cp)
        self.A_day, self.A_slot, self.A_sensor = [np.asarray(f) for f in self.factors]
        # pseudo-inverse so that z = x @ P gives least-squares coefficients on the spatial basis
        self.P = np.linalg.pinv(self.A_sensor).T                      # (N, r_n)
        self.r_n, self.r_s = self.A_sensor.shape[1], self.A_slot.shape[1]
        self.rel_err_ = np.linalg.norm(X3 - self.recon_) / np.linalg.norm(X3)
        return self

    # --- operators ---------------------------------------------------------------
    def project(self, x):
        """(..., N) -> (..., r_n) latent coefficients."""
        return x @ self.P

    def lift(self, z):
        """(..., r_n) -> (..., N) back to sensor space."""
        return z @ self.A_sensor.T

    def slot_embed(self, slot_idx):
        """(...) int -> (..., r_s) time-of-day embedding from the temporal factor."""
        return self.A_slot[slot_idx]

    def n_params(self):
        """Parameters needed to store the learned decomposition."""
        n = sum(f.size for f in self.factors)
        if self.method == "tucker":
            n += self.core.size
        else:
            n += len(self.weights)
        return int(n)

    @staticmethod
    def full_size(X3):
        return int(X3.size)
