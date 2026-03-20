import torch
from torch.distributions import Categorical, kl_divergence
from .metric import Metric
from src.utils.classifier_io import as_pos_prob


class Hubris(Metric):
    def __init__(self, C, dataset_size):
        super().__init__()
        self.C = C
        self.dataset_size = dataset_size
        self.reset()

    @torch.no_grad()
    def update(self, images, batch):
        start_idx, batch_size = batch

        try:
            out = self.C(images, output_feature_maps=True)
        except TypeError:
            out = self.C(images)

        p = as_pos_prob(out)  # (B,)
        self.preds[start_idx:start_idx + batch_size] = p.detach().cpu()

    def _kl(self, p, q):
        return kl_divergence(Categorical(probs=p), Categorical(probs=q)).mean()

    def compute(self, ref_preds=None):
        p = self.preds.clamp(1e-8, 1.0 - 1e-8)
        binary = torch.stack((p, 1.0 - p), dim=1)

        if ref_preds is None:
            ref = torch.full_like(binary, 0.5)
        else:
            rp = ref_preds.clamp(1e-8, 1.0 - 1e-8)
            ref = torch.stack((rp, 1.0 - rp), dim=1)

        worst = torch.linspace(0.0, 1.0, p.size(0), device=p.device).clamp(1e-8, 1.0 - 1e-8)
        worst = torch.stack((worst, 1.0 - worst), dim=1)

        ref_kl = self._kl(worst, ref)
        amb_kl = self._kl(binary, ref)

        return 1.0 - torch.exp(-(amb_kl / (ref_kl + 1e-12)))

    def finalize(self):
        val = self.compute()
        self.result = val.item() if isinstance(val, torch.Tensor) else float(val)
        return self.result

    def get_clfs(self):
        return []

    def reset(self):
        self.result = 1.0
        self.preds = torch.zeros(self.dataset_size, dtype=torch.float32)
