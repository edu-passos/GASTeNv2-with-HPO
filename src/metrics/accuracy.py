import torch


def binary_accuracy(y_pred, y_true, avg=True, threshold=0.5):
    """
    Accepts either probabilities or logits.
    If values are outside [0,1], interpret as logits and apply sigmoid.
    """
    y_pred = y_pred.detach()

    # heuristic: if any value <0 or >1, treat as logits
    if torch.is_tensor(y_pred) and (y_pred.min() < 0 or y_pred.max() > 1):
        y_pred = torch.sigmoid(y_pred)

    correct = (y_pred > threshold) == y_true.bool()
    return correct.sum() if avg is False else correct.float().mean()


def multiclass_accuracy(y_pred, y_true, avg=True):
    pred = y_pred.max(1, keepdim=True)[1]
    correct = pred.eq(y_true.view_as(pred))
    return correct.sum() if avg is False else correct.float().mean()
