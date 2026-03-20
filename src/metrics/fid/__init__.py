import torch
import numpy as np
from .fid_score import calculate_frechet_distance
from .inception import get_inception_feature_map_fn
from .fid_score import load_statistics_from_path, calculate_activation_statistics_dataloader
from ..metric import Metric


class FID(Metric):
    def __init__(self, feature_map_fn, dims, n_images, mu_real, sigma_real, device="cpu", eps=1e-6):
        super().__init__()
        self.feature_map_fn = feature_map_fn
        self.dims = int(dims)
        self.eps = float(eps)
        self.n_images = int(n_images)
        self.mu_real = mu_real
        self.sigma_real = sigma_real
        self.device = device
        self.reset()

    def update(self, images, batch):
        start_idx, batch_size = batch

        with torch.no_grad():
            pred = self.feature_map_fn(images, start_idx, batch_size)

        pred = pred.detach().cpu().numpy()
        n = pred.shape[0]

        # Robust bounds check
        end = start_idx + n
        if end > self.n_images:
            raise RuntimeError(
                f"FID.update out of bounds: start={start_idx}, n={n}, "
                f"end={end} > n_images={self.n_images}"
            )

        self.pred_arr[start_idx:end] = pred
        self.filled[start_idx:end] = True

    def finalize(self):
        if not self.filled.all():
            missing = int((~self.filled).sum())
            raise RuntimeError(f"FID.finalize called but pred_arr has {missing} missing rows.")

        act = self.pred_arr
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)

        self.result = calculate_frechet_distance(mu, sigma, self.mu_real, self.sigma_real, eps=self.eps)
        return self.result

    def reset(self):
        self.pred_arr = np.zeros((self.n_images, self.dims), dtype=np.float64)
        self.filled = np.zeros((self.n_images,), dtype=bool)
        self.result = float("inf")
