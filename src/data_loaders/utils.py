import torch


class BinaryDataset(torch.utils.data.Dataset):
    """
    Lazy binary view over an existing dataset.

    - Keeps only indices of samples whose label ∈ {pos_class, neg_class}
    - Does NOT materialize all images into memory
    - Returns (x, y) where y is int64 {0,1}
    """
    def __init__(self, original_dataset, pos_class, neg_class):
        self.dataset = original_dataset
        self.pos_class = int(pos_class)
        self.neg_class = int(neg_class)

        # Build an index map once (labels are cheap; images are not)
        self.indices = []
        for i in range(len(original_dataset)):
            _, y = original_dataset[i]
            y = int(y) if not torch.is_tensor(y) else int(y.item())
            if y == self.pos_class or y == self.neg_class:
                self.indices.append(i)

        if len(self.indices) == 0:
            raise ValueError(
                f"No samples found for pos={self.pos_class}, neg={self.neg_class}."
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = self.indices[j]
        x, y = self.dataset[i]
        y = int(y) if not torch.is_tensor(y) else int(y.item())
        y_bin = 1 if y == self.pos_class else 0
        return x, torch.tensor(y_bin, dtype=torch.int64)

    @property
    def targets(self):
        # returns a tensor of binary targets aligned with this dataset
        t = []
        for j in range(len(self.indices)):
            _, y = self[j]
            t.append(int(y.item()))
        return torch.tensor(t, dtype=torch.int64)
