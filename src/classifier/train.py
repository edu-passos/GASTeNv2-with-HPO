import argparse
import os
import numpy as np
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data_loaders import load_dataset
from src.metrics.accuracy import multiclass_accuracy
from src.utils import setup_reprod
from src.utils.checkpoint import checkpoint, construct_classifier_from_checkpoint
from src.classifier import construct_classifier  # <- consistent import


def _resolve_device(device_str: str) -> torch.device:
    device_str = (device_str or "cpu").lower()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def evaluate(model, device, dataloader, criterion, acc_fun, verbose=True, desc="Validate", header=None):
    was_training = model.training
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    per_batch_acc = []

    loader = tqdm(dataloader, desc=desc) if verbose else dataloader
    if header:
        print(f"\n--- {header} ---\n")

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            per_batch_acc.append((preds == labels).float().mean().item())

    dataset_size = len(dataloader.dataset)
    epoch_loss = running_loss / max(1, dataset_size)
    epoch_acc = running_corrects / max(1, dataset_size)

    if was_training:
        model.train()

    print(f"{desc}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    print(f"Average batch accuracy: {float(np.mean(per_batch_acc)):.4f}")
    return epoch_acc, epoch_loss


def train(model, optimizer, criterion, train_loader, val_loader, args, name, model_params, device):
    stats = {
        "best_loss": float("inf"),
        "best_epoch": 0,
        "early_stop_counter": 0,
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "val_loss": [],
    }

    cp_path = None
    model.train()

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---\n")
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc="Train"):
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / max(1, len(train_loader.dataset))
        epoch_acc = running_corrects / max(1, len(train_loader.dataset))
        stats["train_loss"].append(epoch_loss)
        stats["train_acc"].append(epoch_acc)

        print(f"Train: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        val_acc, val_loss = evaluate(model, device, val_loader, criterion, multiclass_accuracy)
        stats["val_acc"].append(val_acc)
        stats["val_loss"].append(val_loss)

        if val_loss < stats["best_loss"]:
            stats["best_loss"] = val_loss
            stats["best_epoch"] = epoch
            stats["early_stop_counter"] = 0
            cp_path = checkpoint(model, name, model_params, stats, args, output_dir=args.out_dir)
            print(f"Saved checkpoint: {cp_path}")
        else:
            if args.early_stop is not None:
                stats["early_stop_counter"] += 1
                print(f"Early stop count: {stats['early_stop_counter']}/{args.early_stop}")
                if stats["early_stop_counter"] >= args.early_stop:
                    print("Early stopping triggered")
                    break

    # Ensure we have *some* checkpoint path
    if cp_path is None:
        cp_path = checkpoint(model, name, model_params, stats, args, output_dir=args.out_dir)
        print(f"Saved final checkpoint (no improvement detected): {cp_path}")

    return stats, cp_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("FILESDIR", ".") + "/data", help="Dataset path")
    parser.add_argument("--out-dir", default=os.environ.get("FILESDIR", ".") + "/models", help="Output directory")
    parser.add_argument("--name", default=None, help="Experiment name")
    parser.add_argument("--dataset", dest="dataset_name", default="mnist", help="Dataset name")
    parser.add_argument("--pos", dest="pos_class", type=int, default=0, help="Positive class (binary)")
    parser.add_argument("--neg", dest="neg_class", type=int, default=1, help="Negative class (binary)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--classifier-type", dest="c_type", default="mlp", help="Model architecture")
    parser.add_argument("--nf", type=int, default=512, help="Hidden dim / base features")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--early-stop", type=int, default=None, help="Patience for early stopping")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, cuda:0, ...)")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    seed = args.seed if args.seed is not None else int(np.random.randint(0, 100000))
    setup_reprod(seed)
    print(f"Seed: {seed}")

    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    dataset, num_classes, img_size = load_dataset(
        args.dataset_name,
        args.data_dir,
        pos_class=args.pos_class,
        neg_class=args.neg_class,
    )
    print(f"Dataset: {args.dataset_name}, Classes: {num_classes}, img_size={img_size}")

    # deterministic split
    g = torch.Generator().manual_seed(seed)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model_params = {
        "type": args.c_type,
        "img_size": img_size,
        "nf": args.nf,
        "n_classes": num_classes,
    }
    model = construct_classifier(model_params, device=device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    name = args.name or f"{args.c_type}-{args.nf}-{args.epochs}-{seed}"
    stats, checkpoint_path = train(model, optimizer, criterion, train_loader, val_loader, args, name, model_params, device)

    best_model, *_ = construct_classifier_from_checkpoint(checkpoint_path, device=device)
    print("Evaluating on validation set with best checkpoint...")
    evaluate(best_model, device, val_loader, criterion, multiclass_accuracy, desc="Val(best)", header="Best Model Validation")


if __name__ == "__main__":
    main()
