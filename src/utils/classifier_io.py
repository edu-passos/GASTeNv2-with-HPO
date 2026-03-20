import torch
import torch.nn.functional as F


def as_pos_prob(clf_out):
    """
    Convert classifier output to P(y=pos) in [0,1], shape (B,).

    Supports:
      - (B,2) logits (CrossEntropy classifiers)
      - (B,1) logits or probs
      - (B,)  logits or probs
      - tuples/lists where first element is prediction tensor
    """
    if isinstance(clf_out, (tuple, list)):
        clf_out = clf_out[0]

    if not torch.is_tensor(clf_out):
        raise TypeError(f"Classifier output must be a torch.Tensor (got {type(clf_out)})")

    # (B,2) logits -> softmax -> P(pos)=col 1
    if clf_out.ndim == 2 and clf_out.size(1) == 2:
        return F.softmax(clf_out, dim=1)[:, 1]

    # (B,1) -> (B,)
    if clf_out.ndim == 2 and clf_out.size(1) == 1:
        clf_out = clf_out[:, 0]

    # Now expect (B,)
    if clf_out.ndim != 1:
        raise ValueError(f"Unexpected classifier output shape {tuple(clf_out.shape)}")

    # If already in [0,1], treat as probability; else treat as logit.
    if clf_out.min().item() >= 0.0 and clf_out.max().item() <= 1.0:
        return clf_out
    return torch.sigmoid(clf_out)
