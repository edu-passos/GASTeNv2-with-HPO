import torch
from .metric import Metric
from src.utils.classifier_io import as_pos_prob


class LossSecondTerm(Metric):
    """
    ConfDist metric = mean |0.5 - P(pos)| over generated samples.
    Must match what Step-2 optimizes.
    """
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.reset()

    def update(self, images, batch):
        with torch.no_grad():
            try:
                out = self.C(images, output_feature_maps=True)
            except TypeError:
                out = self.C(images)

            p = as_pos_prob(out)
            term = (0.5 - p).abs().sum().item()

        self.acc += term
        self.count += images.size(0)

    def finalize(self):
        self.result = self.acc / max(1, self.count)
        return self.result

    def reset(self):
        self.count = 0
        self.acc = 0.0
        self.result = float("inf")
