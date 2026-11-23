"""Replace the fine-tuned model's Linear head with an SVM (5-class DR grading) on the frozen backbone features.

1. Features (GPU, once per checkpoint and --tta): every image of train/val/test goes through the fine-tuned
   backbone; the feature right before the Linear head (SwinMS: the concatenated pooled stages) is cached
   in <feat_dir>/features_<split>[_<tta>].npz together with the Linear head's softmax (reference).
   With --tta flip4 the features (and the head softmax) are averaged over the 4 flipped views.
2. SVM: StandardScaler + SVM on the TRAIN features, optionally only --svm_train_devices (e.g. add JYE01,
   whose train images a desktop-only backbone never saw).
   --select cv (default): kernel / C / class_weight chosen by patient-grouped, stratified K-fold CV inside
   TRAIN (score = out-of-fold mtmDR (sens+spec)/2, tie-break quadratic kappa) -> VAL / TEST are never used
   for any choice. --select val: old behaviour, chosen on VAL by kappa (do NOT use when VAL is the test set).
3. Probabilities: multinomial logistic regression on the out-of-fold SVM decision values (cv) -> p0..p4,
   pred = argmax. Written as predictions_<split>.csv in the eval_mtm_vt.py format, so
   eval_mtm_vt_postprocess.py (ALFA) can be run on them.
4. Report (argmax, bootstrap CI) of the Linear head and the SVM on VAL and TEST, overall and per device.
   For DR5_512 (= DR5_512_valpros_dedup): VAL = prospective test (466), TEST = retrospective test (1181).

Example:
  python tools/svm_dr.py --resume <run>/checkpoint-best.pth --data_path /mnt/10TB_2024/Katsu/data/DR/DR5_512 \
      --tta flip4 --svm_train_devices Canon Nidek JYE01
"""
import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo root
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)  # so model / utils import from the repo root

import argparse
import csv
import itertools
import json
import os
import time

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from torchvision import datasets

from model import build_model as _build_model
from utils.metrics import REPORT, TASKS, binary_metrics, bootstrap_ci, load_manifest
from utils.transforms import build_transform

SPLITS = ("train", "val", "test")
TTA_VIEWS = {"none": [()], "flip4": [(), (3,), (2,), (2, 3)]}  # same views as eval_mtm_vt.py


# ---------------------------------------------------------------- features
def build_model(ckpt):
    args = ckpt["args"]
    model = _build_model(args, drop_path_rate=0.0)
    print(model.load_state_dict(ckpt["model"], strict=True))
    return model, args


def embed(model, x):
    """Features right before the classification head."""
    if hasattr(model, "forward_head"):  # timm models / SwinMS: pre_logits = input of the head
        return model.forward_head(model.forward_features(x), pre_logits=True)
    return model.forward_features(x)  # RETFound_mae subclass already returns the pooled feature


@torch.no_grad()
def extract(model, args, split, device, batch_size, num_workers, views):
    ds = datasets.ImageFolder(os.path.join(args.data_path, split), transform=build_transform("val", args))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    feats, probs = [], []
    for x, _ in dl:
        x = x.to(device)
        f_sum, p_sum = 0, 0
        for d in views:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                f = embed(model, torch.flip(x, d) if d else x).float()
                p = torch.softmax(model.head(f).float(), dim=1)
            f_sum, p_sum = f_sum + f, p_sum + p
        feats.append((f_sum / len(views)).cpu())
        probs.append((p_sum / len(views)).cpu())
    names = np.array([os.path.basename(p) for p, _ in ds.samples])
    labels = np.array([int(ds.classes[t]) for _, t in ds.samples])
    return torch.cat(feats).numpy(), torch.cat(probs).numpy(), labels, names


def get_features(cli):
    suffix = "" if cli.tta == "none" else f"_{cli.tta}"
    paths = {s: os.path.join(cli.feat_dir, f"features_{s}{suffix}.npz") for s in SPLITS}
    if all(os.path.exists(p) for p in paths.values()) and not cli.recompute:
        print(f"using cached features in {cli.feat_dir} (tta {cli.tta})")
        out = {}
        for s, p in paths.items():
            d = dict(np.load(p))
            if "head_probs" not in d:  # cache of the old version: logits of one view
                e = np.exp(d["head_logits"] - d["head_logits"].max(1, keepdims=True))
                d["head_probs"] = e / e.sum(1, keepdims=True)
            out[s] = d
        return out

    ckpt = torch.load(cli.resume, map_location="cpu", weights_only=False)
    model, args = build_model(ckpt)
    args.data_path = cli.data_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"checkpoint epoch {ckpt.get('epoch')} | model {args.model} | input {args.input_size} | "
          f"tta {cli.tta} | device {device}")
    os.makedirs(cli.feat_dir, exist_ok=True)
    out = {}
    for s in SPLITS:
        t0 = time.time()
        f, pr, y, n = extract(model, args, s, device, cli.batch_size, cli.num_workers, TTA_VIEWS[cli.tta])
        np.savez(paths[s], feats=f, head_probs=pr, labels=y, names=n)
        out[s] = {"feats": f, "head_probs": pr, "labels": y, "names": n}
        print(f"  {s}: {f.shape} in {time.time() - t0:.0f}s")
    return out


# ---------------------------------------------------------------- SVM
def candidates(kernels, Cs, weights):
    """Cs: kernel -> list of C values"""
    for kernel, cw in itertools.product(kernels, weights):
        for C in Cs[kernel]:
            yield _candidate(kernel, C, cw)


def _candidate(kernel, C, cw):
    cw_ = None if cw == "none" else cw
    if kernel == "linear":
        clf = LinearSVC(C=C, class_weight=cw_, dual="auto", max_iter=20000)
    else:
        clf = SVC(kernel=kernel, C=C, gamma="scale", class_weight=cw_, decision_function_shape="ovr")
    return {"kernel": kernel, "C": C, "class_weight": cw}, make_pipeline(StandardScaler(), clf)


def mtm_bacc(y, pred):
    yb, pb = y >= 2, pred >= 2
    return 0.5 * (pb[yb].mean() + (~pb[~yb]).mean())


def _fold(pipe, X, y, tr, va):
    p = clone(pipe).fit(X[tr], y[tr])
    return va, p.decision_function(X[va])


def oof_decision(pipe, X, y, folds, n_jobs):
    """Out-of-fold decision values (n, 5)."""
    D = np.zeros((len(y), 5))
    for va, d in Parallel(n_jobs=n_jobs)(delayed(_fold)(pipe, X, y, tr, va) for tr, va in folds):
        D[va] = d
    return D


def report_block(title, y, pred, prob, devices, n_boot, rows):
    print(f"\n### {title}")
    print(f"5-class acc {np.mean(pred == y):.4f} | quadratic kappa {cohen_kappa_score(y, pred, weights='quadratic'):.4f}")
    print("confusion matrix (rows=GT 0-4, cols=pred 0-4):\n", confusion_matrix(y, pred, labels=range(5)))
    for task, k in TASKS.items():
        yb, pb = (y >= k).astype(int), (pred >= k).astype(int)
        r = REPORT[task]
        auc = roc_auc_score(yb, prob[:, k:].sum(1)) if 0 < yb.sum() < len(yb) else float("nan")
        print(f"  {task} (pos={yb.sum()}, AUC {auc:.4f})   REPORT acc {r['Accuracy']:.4f} "
              f"sens {r['Sensitivity']:.4f} spec {r['Specificity']:.4f}")
        for grp in ["all"] + sorted(set(devices) - {"?"}):
            mask = np.ones(len(y), bool) if grp == "all" else devices == grp
            if mask.sum() == 0:
                continue
            m = binary_metrics(yb[mask], pb[mask])
            ci = bootstrap_ci(yb[mask], pb[mask], n_boot)
            print(f"    {grp:6} n={mask.sum():4} acc {m['Accuracy']:.4f} sens {m['Sensitivity']:.4f} "
                  f"[{ci['Sensitivity'][0]:.3f},{ci['Sensitivity'][1]:.3f}] spec {m['Specificity']:.4f} "
                  f"[{ci['Specificity'][0]:.3f},{ci['Specificity'][1]:.3f}]  TP{m['TP']} FN{m['FN']} TN{m['TN']} FP{m['FP']}")
            rows.append([title, task, grp, int(mask.sum()), int(yb[mask].sum()), m["Accuracy"], *ci["Accuracy"],
                         m["Sensitivity"], *ci["Sensitivity"], m["Specificity"], *ci["Specificity"],
                         m["TP"], m["FN"], m["TN"], m["FP"]])


def write_predictions(path, names, y, prob, manifest):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "cohort", "device", "label", "pred", "p0", "p1", "p2", "p3", "p4"])
        for n, yy, pp in zip(names, y, prob):
            m = manifest.get(n, {})
            w.writerow([n, m.get("cohort", ""), m.get("device", ""), int(yy), int(pp.argmax()), *[f"{v:.5f}" for v in pp]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True, help="fine-tuned checkpoint-best.pth")
    ap.add_argument("--data_path", default="Fundus_Dataset/DR5_512_prosval")
    ap.add_argument("--tta", default="none", choices=list(TTA_VIEWS), help="average the features over flipped views")
    ap.add_argument("--feat_dir", default=None, help="feature cache, defaults to <checkpoint folder>/svm_features")
    ap.add_argument("--out_dir", default=None,
                    help="results, defaults to <checkpoint folder>/svm_<tta>_<select>_<svm_train_devices>")
    ap.add_argument("--svm_train_devices", nargs="+", default=None,
                    help="train the SVM only on TRAIN images of these devices (manifest.csv); default all")
    ap.add_argument("--select", default="cv", choices=["cv", "val"],
                    help="cv = patient-grouped K-fold inside TRAIN (default); val = choose on VAL (old behaviour)")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--recompute", action="store_true", help="re-extract features even if cached")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--n_jobs", type=int, default=16, help="CPU processes for the CV fits")
    ap.add_argument("--kernels", nargs="+", default=["linear", "rbf"])
    # fine-tuned features are nearly separable on TRAIN: strong regularisation (small C) wins for linear,
    # and a large linear C is slow (minutes per fit)
    ap.add_argument("--linear_Cs", nargs="+", type=float, default=[0.0001, 0.0003, 0.001, 0.003, 0.01])
    ap.add_argument("--rbf_Cs", nargs="+", type=float, default=[0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--class_weights", nargs="+", default=["none", "balanced"])
    ap.add_argument("--n_boot", type=int, default=2000)
    cli = ap.parse_args()
    ck_dir = os.path.dirname(os.path.abspath(cli.resume))
    cli.feat_dir = cli.feat_dir or os.path.join(ck_dir, "svm_features")
    devs_tag = "".join(cli.svm_train_devices) if cli.svm_train_devices else "all"
    cli.out_dir = cli.out_dir or os.path.join(ck_dir, f"svm_{cli.tta}_{cli.select}_{devs_tag}")
    os.makedirs(cli.out_dir, exist_ok=True)

    F = get_features(cli)
    manifest = load_manifest(cli.data_path)
    dev = {s: np.array([manifest.get(n, {}).get("device", "?") for n in F[s]["names"]]) for s in SPLITS}
    keep = np.isin(dev["train"], cli.svm_train_devices) if cli.svm_train_devices else np.ones(len(dev["train"]), bool)
    Xtr, ytr = F["train"]["feats"][keep], F["train"]["labels"][keep]
    groups = np.array([manifest.get(n, {}).get("patient", n) for n in F["train"]["names"][keep]])
    print(f"\nSVM train: {keep.sum()}/{len(keep)} TRAIN images (devices {cli.svm_train_devices or 'all'}), "
          f"{len(set(groups))} patients, per class {np.bincount(ytr, minlength=5).tolist()}")

    # ---- model selection
    grid_rows, best = [], None
    if cli.select == "cv":
        folds = list(StratifiedGroupKFold(cli.cv_folds, shuffle=True, random_state=0).split(Xtr, ytr, groups))
        print(f"grid -> {cli.cv_folds}-fold patient-grouped CV inside TRAIN, score = OOF mtmDR (sens+spec)/2, "
              f"tie-break quadratic kappa")
        print(f"{'kernel':7} {'C':>7} {'class_w':9} {'oof_mtm_bacc':>12} {'oof_qwk':>8} {'oof_acc':>8} {'sec':>5}")
        for params, pipe in candidates(cli.kernels, {'linear': cli.linear_Cs, 'rbf': cli.rbf_Cs}, cli.class_weights):
            t0 = time.time()
            D = oof_decision(pipe, Xtr, ytr, folds, cli.n_jobs)
            pv = D.argmax(1)
            sc = (mtm_bacc(ytr, pv), cohen_kappa_score(ytr, pv, weights="quadratic"))
            print(f"{params['kernel']:7} {params['C']:7g} {params['class_weight']:9} {sc[0]:12.4f} {sc[1]:8.4f} "
                  f"{np.mean(pv == ytr):8.4f} {time.time() - t0:5.0f}", flush=True)
            grid_rows.append([params["kernel"], params["C"], params["class_weight"], sc[0], sc[1], np.mean(pv == ytr)])
            if best is None or sc > best[0]:
                best = (sc, params, pipe, D)
        calib_X = best[3]  # out-of-fold decision values -> honest calibration
        header = ["kernel", "C", "class_weight", "oof_mtm_bacc", "oof_qwk", "oof_acc"]
    else:
        Xva, yva = F["val"]["feats"], F["val"]["labels"]
        print("grid -> chosen on VAL by kappa (only valid if VAL is not the test set)")
        for params, pipe in candidates(cli.kernels, {'linear': cli.linear_Cs, 'rbf': cli.rbf_Cs}, cli.class_weights):
            pv = clone(pipe).fit(Xtr, ytr).predict(Xva)
            sc = (cohen_kappa_score(yva, pv), mtm_bacc(yva, pv))
            print(f"{params['kernel']:7} {params['C']:7g} {params['class_weight']:9} val kappa {sc[0]:.4f}", flush=True)
            grid_rows.append([params["kernel"], params["C"], params["class_weight"], sc[0], sc[1]])
            if best is None or sc > best[0]:
                best = (sc, params, pipe, None)
        header = ["kernel", "C", "class_weight", "val_kappa", "val_mtm_bacc"]
    with open(os.path.join(cli.out_dir, f"svm_grid_{cli.select}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(grid_rows)
    params = best[1]
    print(f"\nselected: {params} (score {tuple(round(float(s), 4) for s in best[0])})")

    # ---- final SVM on all selected TRAIN images + probability calibration
    final = clone(best[2]).fit(Xtr, ytr)
    if best[3] is None:
        calib_X = final.decision_function(Xtr)  # in-sample (select val): over-confident
    calib = LogisticRegression(max_iter=5000).fit(calib_X, ytr)
    prob = {s: calib.predict_proba(final.decision_function(F[s]["feats"])) for s in ("val", "test")}

    rows = []
    for s, desc in (("val", "VAL"), ("test", "TEST")):
        y = F[s]["labels"]
        hp = F[s]["head_probs"]
        print(f"\n================ {desc} (n={len(y)}) ================")
        report_block(f"{desc} Linear head (fine-tuned, tta {cli.tta})", y, hp.argmax(1), hp, dev[s], cli.n_boot, rows)
        report_block(f"{desc} SVM {params['kernel']} C={params['C']:g} cw={params['class_weight']}",
                     y, prob[s].argmax(1), prob[s], dev[s], cli.n_boot, rows)
        write_predictions(os.path.join(cli.out_dir, f"predictions_{s}.csv"), F[s]["names"], y, prob[s], manifest)

    with open(os.path.join(cli.out_dir, "svm_mtm_vt_metrics.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["classifier", "task", "group", "n", "pos", "acc", "acc_lo", "acc_hi", "sens", "sens_lo", "sens_hi",
                    "spec", "spec_lo", "spec_hi", "TP", "FN", "TN", "FP"])
        w.writerows(rows)
    with open(os.path.join(cli.out_dir, "svm_config.json"), "w", encoding="utf-8") as fh:
        json.dump({**{k: v for k, v in vars(cli).items()}, "selected": params,
                   "selection_score": [float(s) for s in best[0]], "n_train": int(keep.sum())}, fh, indent=1)
    print(f"\nsaved predictions_val.csv / predictions_test.csv (eval_mtm_vt format: run eval_mtm_vt_postprocess.py on "
          f"them), svm_grid_{cli.select}.csv, svm_mtm_vt_metrics.csv, svm_config.json to {cli.out_dir}")


if __name__ == "__main__":
    main()
