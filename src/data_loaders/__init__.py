import torch
from .datasets import get_mnist, get_fashion_mnist, get_cifar10, get_stl10, get_chest_xray, get_imagenet
from .utils import BinaryDataset

dataset_2_fn = {
    'mnist': get_mnist,
    'fashion-mnist': get_fashion_mnist,
    'cifar10': get_cifar10,
    'stl10': get_stl10,
    'chest-xray': get_chest_xray,
    'imagenet': get_imagenet,
}

def valid_dataset(name):
    return name.lower() in dataset_2_fn

def load_dataset(name, data_dir, pos_class=None, neg_class=None, train=True):
    if not valid_dataset(name):
        print("{} dataset not supported".format(name))
        exit(-1)

    get_dset_fn = dataset_2_fn[name.lower()]
    dataset = get_dset_fn(data_dir, train=train)

    sample, _ = dataset[0]
    image_size = sample.shape
    if len(image_size) == 2:
        image_size = (1, *image_size)
    elif len(image_size) == 3:
        if image_size[-1] == 3:
            image_size = (image_size[-1],) + image_size[:-1]

    # Try to get targets from the dataset.
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        targets = targets if torch.is_tensor(targets) else torch.as_tensor(targets)
        targets = targets.view(-1).to(torch.long)
        num_classes = int(targets.unique().numel())
    else:
        num_classes = None


    if pos_class is not None and neg_class is not None:
        num_classes = 2
        dataset = BinaryDataset(dataset, pos_class, neg_class)

    return dataset, num_classes, image_size
