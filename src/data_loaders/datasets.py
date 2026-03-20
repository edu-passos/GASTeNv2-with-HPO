from pathlib import Path
import os
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
from datasets import load_dataset as hf_load_dataset
from PIL import Image


def _norm(mean, std):
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


# ─── torchvision datasets ────────────────────────────────────────────────
def get_mnist(root, train=True):
    return torchvision.datasets.MNIST(
        root, download=True, train=train,
        transform=_norm((0.5,), (0.5,))
    )


def get_fashion_mnist(root, train=True):
    return torchvision.datasets.FashionMNIST(
        root, download=True, train=train,
        transform=_norm((0.5,), (0.5,))
    )


def get_cifar10(root, train=True):
    return torchvision.datasets.CIFAR10(
        root, download=True, train=train,
        transform=_norm((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
    )


def get_stl10(root, train=True):
    split = "train" if train else "test"
    return torchvision.datasets.STL10(
        root, download=True, split=split,
        transform=_norm((0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5))
    )


# ─── Chest-XRay (HF Hub) ────────────────────────────────────────────────
def get_chest_xray(root, train=True):
    split = "train" if train else "test"
    ds = hf_load_dataset(
        "keremberke/chest-xray-classification", "full",
        split=split, cache_dir=root
    )

    tfm = T.Compose([
        T.Lambda(lambda im: im.convert("L")),   # <-- 1-channel grayscale
        T.Resize((128, 128)),
        _norm((0.5,), (0.5,))                  # <-- 1-channel norm to [-1,1]
    ])

    class CXR(Dataset):
        def __init__(self, d): self.d = d
        def __len__(self): return len(self.d)
        def __getitem__(self, i):
            x = tfm(self.d[i]["image"])
            y = self.d[i]["labels"]
            return x, y
        @property
        def targets(self):
            return torch.tensor([self.d[i]["labels"] for i in range(len(self.d))])

    return CXR(ds)


# ─── ImageNet-1k ───────────────────────────────
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")


def get_imagenet(root, train=True, *, streaming=False, use_local_folders=False):
    split = "train" if train else "validation"

    if use_local_folders:
        folder = Path(root) / ("train" if train else "val")
        ds = hf_load_dataset("imagefolder",
                             data_dir=str(folder),
                             split="train")
    else:
        ds = hf_load_dataset(
            "imagenet-1k",
            split=split,
            cache_dir=root,
            streaming=streaming,
            token=HF_TOKEN,
            trust_remote_code=True
        )

    tfm = T.Compose([
        T.Resize((224, 224)),
        _norm((0.485, 0.456, 0.406),
              (0.229, 0.224, 0.225))
    ])

    class IM(Dataset):
        def __init__(self, d): self.d = d
        def __len__(self): return len(self.d)
        def __getitem__(self, i):
            ex  = self.d[i]
            img = ex["image"]
            if isinstance(img, Image.Image):
                img = img.convert("RGB")
            x = tfm(img)
            return x, ex["label"]

        @property
        def targets(self):
            return torch.tensor(self.d["label"])

    return IM(ds)


# ─── registry helper ────────────────────────────────────────────────────
FACTORY = {
    "mnist":          get_mnist,
    "fashion-mnist":  get_fashion_mnist,
    "cifar10":        get_cifar10,
    "stl10":          get_stl10,
    "chestxray":      get_chest_xray,
    "imagenet":       get_imagenet,
}


def load_dataset(name, root, **kw):
    if name not in FACTORY:
        raise KeyError(name)
    return FACTORY[name](root, **kw)
