"""ImageFolder datasets: <data_path>/{train,val,test}/<class>/*.png"""
import os

import torch
from torch.utils.data import Subset
from torchvision import datasets

from utils.transforms import build_transform


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)

    # --train_devices / --val_devices: keep only images of these devices (from <data_path>/manifest.csv);
    # TEST is never filtered
    keep = {"train": getattr(args, "train_devices", None), "val": getattr(args, "val_devices", None)}.get(is_train)
    if keep:
        _filter_devices(dataset, args.data_path, keep, is_train)

    if is_train == 'train':
        ratio = float(getattr(args, "dataratio", 1.0))
        seed = int(getattr(args, "seed", 0))
        stratified = bool(getattr(args, "stratified", False))

        if 0.0 < ratio < 1.0:
            if stratified:
                idx = _stratified_indices(dataset.targets, ratio, seed)
            else:
                # simple uniform subsample with torch.Generator for reproducibility
                g = torch.Generator().manual_seed(seed)
                n = len(dataset)
                k = max(1, int(n * ratio))
                idx = torch.randperm(n, generator=g)[:k].tolist()
            dataset = Subset(dataset, idx)

    return dataset


def _filter_devices(dataset, data_path, devices, split):
    """In-place: keep only images whose device (from <data_path>/manifest.csv) is in `devices`."""
    import csv
    from collections import Counter
    path = os.path.join(data_path, "manifest.csv")
    with open(path, encoding="utf-8") as fh:
        device_of = {r["image"]: r["device"] for r in csv.DictReader(fh)}
    missing = [p for p, _ in dataset.samples if os.path.basename(p) not in device_of]
    if missing:
        raise ValueError(f"{len(missing)} {split} images not in {path}, e.g. {missing[0]}")
    n0 = len(dataset.samples)
    dataset.samples = [(p, t) for p, t in dataset.samples if device_of[os.path.basename(p)] in devices]
    dataset.imgs = dataset.samples
    dataset.targets = [t for _, t in dataset.samples]
    if not dataset.samples:
        raise ValueError(f"no {split} images left for devices {devices}")
    cnt = Counter(dataset.targets)
    print(f"[devices] {split}: kept {len(dataset.samples)}/{n0} images of {devices}; "
          f"per class {[cnt.get(i, 0) for i in range(len(dataset.classes))]}")


def _stratified_indices(targets, ratio: float, seed: int):
    """Maintain class proportions. Ensures at least 1 sample per class when possible."""
    t = torch.as_tensor(targets)
    classes = torch.unique(t)
    g = torch.Generator().manual_seed(seed)

    keep = []
    for c in classes.tolist():
        cls_idx = torch.nonzero(t == c, as_tuple=False).view(-1)
        if len(cls_idx) == 0:
            continue
        k = max(1, int(round(len(cls_idx) * ratio)))
        sel = cls_idx[torch.randperm(len(cls_idx), generator=g)[:k]]
        keep.extend(sel.tolist())

    # shuffle final indices (stable across seed)
    g2 = torch.Generator().manual_seed(seed + 1)
    keep = torch.tensor(keep)[torch.randperm(len(keep), generator=g2)].tolist()
    return keep
