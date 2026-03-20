import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor
import PIL

from .metric import Metric
from .hubris import Hubris


class OutputsHistogram(Metric):
    """
    Collects classifier output probabilities (positive class) across the evaluation set
    and provides plotting utilities that are safe in headless training (no plt.show()).

    Works with:
      - binary logits shape [B] or [B,1]  -> sigmoid
      - multiclass logits shape [B,K]     -> softmax -> take class 1 prob by convention
      - feature-map API: C(x, output_feature_maps=True) returning tuple/list with logits at index 0
    """
    def __init__(self, C, dataset_size: int):
        super().__init__()
        self.C = C
        self.dataset_size = int(dataset_size)
        self.y_hat = torch.zeros(self.dataset_size, dtype=torch.float32)
        self.to_tensor = ToTensor()
        self.hubris = Hubris(C, self.dataset_size)

    def _clf_logits(self, images: torch.Tensor) -> torch.Tensor:
        # Prefer feature-map API if available; else plain forward
        try:
            out = self.C(images, output_feature_maps=True)
            logits = out[0]  # convention used elsewhere in your repo
        except TypeError:
            logits = self.C(images)
        return logits

    @staticmethod
    def _to_pos_prob(logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probability for "positive class".
        Convention:
          - if logits is [B, K] with K>=2 -> use softmax[:,1]
          - else -> sigmoid(logits)
        """
        if logits.ndim == 2 and logits.size(1) >= 2:
            return F.softmax(logits, dim=1)[:, 1]
        return torch.sigmoid(logits.view(-1))

    @torch.no_grad()
    def update(self, images, batch):
        start, bs = batch
        start = int(start)
        bs = int(bs)

        # keep hubris in sync
        self.hubris.update(images, batch)

        logits = self._clf_logits(images)
        probs = self._to_pos_prob(logits)

        # store on CPU
        self.y_hat[start:start + bs] = probs.detach().float().cpu()

    def _hist_plot(self, data: np.ndarray, ax, title: str, bins: int = 30, xlim=None):
        # Always use histogram (no seaborn dependency; stable headless)
        ax.hist(data, bins=bins, density=True)
        ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(*xlim)

    def plot(self):
        """
        Returns a torch.Tensor image (CHW) of a single histogram plot.
        Does NOT call plt.show().
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        data = self.y_hat.numpy()
        self._hist_plot(data, ax, "Classifier Output Distribution", bins=30, xlim=(0, 1))
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Density")

        fig.canvas.draw()
        buf, (w, h) = fig.canvas.print_to_buffer()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        img = PIL.Image.fromarray(arr, mode="RGBA").convert("RGB")
        plt.close(fig)
        return self.to_tensor(img)

    def plot_clfs(self):
        """
        Returns a torch.Tensor image (CHW) with:
          1) output probability distribution
          2) confusion distance distribution |p-0.5|
          3) hubris scalar bar
        """
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        p = self.y_hat.numpy()
        self._hist_plot(p, axs[0], "Classifier Output Distribution", bins=30, xlim=(0, 1))

        cd = np.abs(0.5 - p)
        self._hist_plot(cd, axs[1], "Confusion Distance Distribution", bins=30, xlim=(0, 0.5))

        hubris_val = float(self.hubris.finalize())
        axs[2].bar(["Hubris"], [hubris_val])
        axs[2].set_ylim(0, 1)
        axs[2].set_title("Hubris")

        fig.tight_layout()
        fig.canvas.draw()

        buf, (w, h) = fig.canvas.print_to_buffer()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        img = PIL.Image.fromarray(arr, mode="RGBA").convert("RGB")
        plt.close(fig)
        return self.to_tensor(img)

    def reset(self):
        self.y_hat.zero_()
        self.hubris.reset()
