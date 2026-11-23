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


def _device_map(data_path):
    """image file name -> device, from <data_path>/manifest.csv"""
    import csv
    path = os.path.join(data_path, "manifest.csv")
    with open(path, encoding="utf-8") as fh:
        return {r["image"]: r["device"] for r in csv.DictReader(fh)}


def device_class_weights(dataset, data_path):
    """--balance device_class: per-sample weights for a WeightedRandomSampler.

    w(device d, class c) = p(c) / p(c | d), with p(c) = class distribution of the whole TRAIN set.
    In the sampled stream every device then has the same class distribution p(c) (the grade no longer
    depends on the device, e.g. JYE01 is no longer mostly grade 0), while every device keeps its share of
    the samples (sum of the weights of a device = its number of images).
    """
    from collections import Counter
    base, idx = (dataset.dataset, dataset.indices) if isinstance(dataset, Subset) else (dataset, range(len(dataset)))
    device_of = _device_map(data_path)
    keys = []
    for i in idx:
        p, t = base.samples[i]
        name = os.path.basename(p)
        if name not in device_of:
            raise ValueError(f"--balance device_class: {name} not in {data_path}/manifest.csv")
        keys.append((device_of[name], t))
    n = len(keys)
    n_c = Counter(t for _, t in keys)
    n_d = Counter(d for d, _ in keys)
    n_dc = Counter(keys)
    w_dc = {(d, c): (n_c[c] / n) / (n_dc[(d, c)] / n_d[d]) for (d, c) in n_dc}
    print("[balance] device_class: target class distribution "
          f"{[round(n_c.get(c, 0) / n, 3) for c in range(len(base.classes))]}")
    for d in sorted(n_d):
        print(f"[balance]   {d:6} n={n_d[d]:5}  per class {[n_dc.get((d, c), 0) for c in range(len(base.classes))]}"
              f"  weight {[round(w_dc[(d, c)], 2) if (d, c) in w_dc else '-' for c in range(len(base.classes))]}")
    missing = [(d, c) for d in n_d for c in n_c if (d, c) not in n_dc]
    if missing:
        print(f"[balance]   warning: no images for (device, class) {missing}; those cells cannot be balanced")
    return [w_dc[k] for k in keys]


def _filter_devices(dataset, data_path, devices, split):
    """In-place: keep only images whose device (from <data_path>/manifest.csv) is in `devices`."""
    from collections import Counter
    path = os.path.join(data_path, "manifest.csv")
    device_of = _device_map(data_path)
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
