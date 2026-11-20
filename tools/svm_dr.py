"""Replace the fine-tuned model's Linear head with an SVM (5-class DR grading).

1. Feature extraction (GPU, once): every image of train/val/test goes through the fine-tuned
   backbone; the 1024-d CLS token after the final LayerNorm (= the exact input of the Linear head)
   is cached to <out_dir>/features_<split>.npz. Later runs reuse the cache (no GPU needed).
2. SVM: StandardScaler + SVM trained on the TRAIN features (5 classes).
   Hyper-parameters (kernel, C, class_weight) are chosen on the VAL split by Cohen's kappa.
   The chosen model is evaluated once on TEST.
3. The 5-class predictions are mapped to mtmDR (grade>=2) and vtDR (grade>=3) and reported with
   bootstrap 95% CIs, overall and per device, next to the original Linear head and the original paper.

Example:
  python tools/svm_dr.py --resume output_dir/DR5_shanghai_224_v2/checkpoint-best.pth --data_path Fundus_Dataset/DR5_512_prosval
"""
import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo root
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)  # so model / utils import from the repo root

import argparse
import csv
import itertools
import os
import time

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from torchvision import datasets

from model import build_model as _build_model
from utils.metrics import REPORT, TASKS, binary_metrics, bootstrap_ci, load_manifest
from utils.transforms import build_transform

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- features
def build_model(ckpt):
    args = ckpt["args"]
    model = _build_model(args, drop_path_rate=0.0)
    print(model.load_state_dict(ckpt["model"], strict=True))
    return model, args


def embed(model, x):
    """Features right before the classification head."""
    if hasattr(model, "forward_head"):  # timm ViT (RETFound_dinov2 / Dinov2): CLS token after final norm
        return model.forward_head(model.forward_features(x), pre_logits=True)
    return model.forward_features(x)  # RETFound_mae subclass already returns the pooled feature


@torch.no_grad()
def extract(model, args, split, device, batch_size, num_workers):
    ds = datasets.ImageFolder(os.path.join(args.data_path, split), transform=build_transform("val", args))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    feats, logits = [], []
    for x, _ in dl:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            f = embed(model, x.to(device)).float()
            logits.append(model.head(f).float().cpu())
        feats.append(f.cpu())
    names = np.array([os.path.basename(p) for p, _ in ds.samples])
    labels = np.array([int(ds.classes[t]) for _, t in ds.samples])
    return torch.cat(feats).numpy(), torch.cat(logits).numpy(), labels, names


def get_features(cli):
    paths = {s: os.path.join(cli.out_dir, f"features_{s}.npz") for s in SPLITS}
    if all(os.path.exists(p) for p in paths.values()) and not cli.recompute:
        print(f"using cached features in {cli.out_dir}")
        return {s: dict(np.load(p)) for s, p in paths.items()}

    ckpt = torch.load(cli.resume, map_location="cpu", weights_only=False)
    model, args = build_model(ckpt)
    args.data_path = cli.data_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"checkpoint epoch {ckpt.get('epoch')} | model {args.model} | input {args.input_size} | "
          f"aug_mode {getattr(args, 'aug_mode', 'default')} | device {device}")
    out = {}
    for s in SPLITS:
        t0 = time.time()
        f, lg, y, n = extract(model, args, s, device, cli.batch_size, cli.num_workers)
        np.savez(paths[s], feats=f, head_logits=lg, labels=y, names=n)
        out[s] = {"feats": f, "head_logits": lg, "labels": y, "names": n}
        print(f"  {s}: {f.shape} in {time.time() - t0:.0f}s")
    return out


# ---------------------------------------------------------------- SVM
def candidates(kernels, Cs, weights):
    for kernel, C, cw in itertools.product(kernels, Cs, weights):
        cw_ = None if cw == "none" else cw
        if kernel == "linear":
            clf = LinearSVC(C=C, class_weight=cw_, dual="auto", max_iter=20000)
        else:
            clf = SVC(kernel=kernel, C=C, gamma="scale", class_weight=cw_, decision_function_shape="ovr")
        yield {"kernel": kernel, "C": C, "class_weight": cw}, make_pipeline(StandardScaler(), clf)


def report_block(title, y, pred, devices, n_boot, rows):
    print(f"\n### {title}")
    print(f"5-class acc {np.mean(pred == y):.4f} | kappa {cohen_kappa_score(y, pred):.4f} | "
          f"quadratic kappa {cohen_kappa_score(y, pred, weights='quadratic'):.4f}")
    print("confusion matrix (rows=GT 0-4, cols=pred 0-4):\n", confusion_matrix(y, pred, labels=range(5)))
    for task, k in TASKS.items():
        yb, pb = (y >= k).astype(int), (pred >= k).astype(int)
        r = REPORT[task]
        print(f"  {task} (pos={yb.sum()})   REPORT acc {r['Accuracy']:.4f} sens {r['Sensitivity']:.4f} spec {r['Specificity']:.4f}")
        for grp in ["all", "JYE01", "Canon", "Nidek"]:
            mask = np.ones(len(y), bool) if grp == "all" else devices == grp
            if mask.sum() == 0:
                continue
            m = binary_metrics(yb[mask], pb[mask])
            ci = bootstrap_ci(yb[mask], pb[mask], n_boot)
            print(f"    {grp:6} n={mask.sum():3} acc {m['Accuracy']:.4f} sens {m['Sensitivity']:.4f} "
                  f"[{ci['Sensitivity'][0]:.3f},{ci['Sensitivity'][1]:.3f}] spec {m['Specificity']:.4f} "
                  f"[{ci['Specificity'][0]:.3f},{ci['Specificity'][1]:.3f}]  TP{m['TP']} FN{m['FN']} TN{m['TN']} FP{m['FP']}")
            rows.append([title, task, grp, int(mask.sum()), int(yb[mask].sum()), m["Accuracy"], *ci["Accuracy"],
                         m["Sensitivity"], *ci["Sensitivity"], m["Specificity"], *ci["Specificity"],
                         m["TP"], m["FN"], m["TN"], m["FP"]])
        beat = all(binary_metrics(yb, pb)[s] >= r[s] for s in ("Sensitivity", "Specificity"))
        print(f"    -> beats report on Sens & Spec: {'YES' if beat else 'no'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True, help="fine-tuned checkpoint-best.pth")
    ap.add_argument("--data_path", default="Fundus_Dataset/DR5_512_prosval")
    ap.add_argument("--out_dir", default=None, help="defaults to <checkpoint folder>/svm")
    ap.add_argument("--recompute", action="store_true", help="re-extract features even if cached")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--kernels", nargs="+", default=["linear", "rbf"])
    ap.add_argument("--Cs", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--class_weights", nargs="+", default=["none", "balanced"])
    ap.add_argument("--n_boot", type=int, default=2000)
    cli = ap.parse_args()
    cli.out_dir = cli.out_dir or os.path.join(os.path.dirname(os.path.abspath(cli.resume)), "svm")
    os.makedirs(cli.out_dir, exist_ok=True)

    F = get_features(cli)
    Xtr, ytr = F["train"]["feats"], F["train"]["labels"]
    Xva, yva = F["val"]["feats"], F["val"]["labels"]
    Xte, yte = F["test"]["feats"], F["test"]["labels"]

    # sanity: the cached head logits must reproduce the Linear head's argmax
    head_pred_te = F["test"]["head_logits"].argmax(1)
    head_pred_va = F["val"]["head_logits"].argmax(1)
    print(f"\nLinear head (reference): val kappa {cohen_kappa_score(yva, head_pred_va):.4f} | "
          f"test acc {np.mean(head_pred_te == yte):.4f}")

    # ---- model selection on val
    print(f"\nSVM grid on train ({len(ytr)}) -> select on val ({len(yva)}) by kappa")
    print(f"{'kernel':7} {'C':>7} {'class_w':9} {'train_acc':>9} {'val_acc':>8} {'val_kappa':>9} {'val_f1':>7} {'sec':>5}")
    grid_rows, best = [], None
    for params, pipe in candidates(cli.kernels, cli.Cs, cli.class_weights):
        t0 = time.time()
        pipe.fit(Xtr, ytr)
        tr_acc = np.mean(pipe.predict(Xtr) == ytr)
        pv = pipe.predict(Xva)
        kappa, f1 = cohen_kappa_score(yva, pv), f1_score(yva, pv, average="macro", zero_division=0)
        sec = time.time() - t0
        print(f"{params['kernel']:7} {params['C']:7g} {params['class_weight']:9} {tr_acc:9.4f} "
              f"{np.mean(pv == yva):8.4f} {kappa:9.4f} {f1:7.4f} {sec:5.0f}", flush=True)
        grid_rows.append([params["kernel"], params["C"], params["class_weight"], tr_acc, np.mean(pv == yva), kappa, f1])
        if best is None or (kappa, f1) > (best[0], best[1]):
            best = (kappa, f1, params, pipe)
    with open(os.path.join(cli.out_dir, "svm_grid_val.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kernel", "C", "class_weight", "train_acc", "val_acc", "val_kappa", "val_macro_f1"])
        w.writerows(grid_rows)
    print(f"\nselected: {best[2]} (val kappa {best[0]:.4f}, macro F1 {best[1]:.4f})")

    # ---- test once
    svm_pred = best[3].predict(Xte)
    manifest = load_manifest(cli.data_path)
    devices = np.array([manifest.get(n, {}).get("device", "?") for n in F["test"]["names"]])
    rows = []
    report_block("Linear head (fine-tuned)", yte, head_pred_te, devices, cli.n_boot, rows)
    report_block(f"SVM {best[2]['kernel']} C={best[2]['C']:g} cw={best[2]['class_weight']}", yte, svm_pred, devices, cli.n_boot, rows)

    with open(os.path.join(cli.out_dir, "svm_predictions_test.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "device", "label", "head_pred", "svm_pred"])
        w.writerows(zip(F["test"]["names"], devices, yte, head_pred_te, svm_pred))
    with open(os.path.join(cli.out_dir, "svm_mtm_vt_metrics.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["classifier", "task", "group", "n", "pos", "acc", "acc_lo", "acc_hi", "sens", "sens_lo", "sens_hi",
                    "spec", "spec_lo", "spec_hi", "TP", "FN", "TN", "FP"])
        w.writerows(rows)
    print(f"\nsaved features_*.npz, svm_grid_val.csv, svm_predictions_test.csv, svm_mtm_vt_metrics.csv to {cli.out_dir}")


if __name__ == "__main__":
    main()
