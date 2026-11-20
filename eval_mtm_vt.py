"""Evaluate a fine-tuned 5-class DR model on mtmDR / vtDR, compared with the original paper.

mtmDR: grades {0,1} vs {2,3,4}   score = P(2)+P(3)+P(4)
vtDR : grades {0,1,2} vs {3,4}   score = P(3)+P(4)

Two operating points are reported for each task:
  argmax  : positive if the 5-class argmax falls in the positive grades
  val-thr : threshold on the summed probability, tuned on the prospective part of val
            (Youden J); the test set is never used for tuning.
All metrics come with bootstrap 95% CIs, overall and per device.

Example:
  python eval_mtm_vt.py --resume output_dir/<task>/checkpoint-best.pth --data_path Fundus_Dataset/DR5_512_prosval
"""
import argparse
import csv
import os

import numpy as np
import torch
from torchvision import datasets

from model import build_model
from utils.metrics import REPORT, TASKS, binary_metrics, bootstrap_ci, load_manifest, youden_threshold
from utils.transforms import build_transform


@torch.no_grad()
def predict(model, args, split, device):
    ds = datasets.ImageFolder(os.path.join(args.data_path, split), transform=build_transform("val", args))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    probs = []
    for x, _ in dl:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probs.append(torch.softmax(model(x.to(device)).float(), dim=1).cpu())
    names = [os.path.basename(p) for p, _ in ds.samples]
    labels = np.array([int(ds.classes[t]) for _, t in ds.samples])
    return names, labels, torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True, help="checkpoint-best.pth from train_dr.py (or RETFound main_finetune.py)")
    ap.add_argument("--data_path", default="Fundus_Dataset/DR5_512_prosval")
    ap.add_argument("--out_dir", default=None, help="defaults to the checkpoint's folder")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--n_boot", type=int, default=2000)
    cli = ap.parse_args()

    ckpt = torch.load(cli.resume, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    args.data_path, args.batch_size, args.num_workers = cli.data_path, cli.batch_size, cli.num_workers
    out_dir = cli.out_dir or os.path.dirname(os.path.abspath(cli.resume))

    model = build_model(args, drop_path_rate=0.0)
    print(model.load_state_dict(ckpt["model"], strict=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"checkpoint epoch {ckpt.get('epoch')} | model {args.model} | input {args.input_size}")

    manifest = load_manifest(cli.data_path)
    device_of = lambda n: manifest.get(n, {}).get("device", "?")
    cohort_of = lambda n: manifest.get(n, {}).get("cohort", "?")

    res = {}
    for split in ("val", "test"):
        names, y, p = predict(model, args, split, device)
        res[split] = (names, y, p)
        with open(os.path.join(out_dir, f"predictions_{split}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "cohort", "device", "label", "pred", "p0", "p1", "p2", "p3", "p4"])
            for n, yy, pp in zip(names, y, p):
                w.writerow([n, cohort_of(n), device_of(n), yy, int(pp.argmax()), *[f"{v:.5f}" for v in pp]])

    # 5-class summary on test
    names, y, p = res["test"]
    pred5 = p.argmax(1)
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    print(f"\n[test 5-class] acc {np.mean(pred5 == y):.4f} | quadratic kappa {cohen_kappa_score(y, pred5, weights='quadratic'):.4f}")
    print("confusion matrix (rows=GT 0-4, cols=pred 0-4):\n", confusion_matrix(y, pred5, labels=range(5)))

    vnames, vy, vp = res["val"]
    pros_val = np.array([cohort_of(n) == "pros" for n in vnames])
    devices = np.array([device_of(n) for n in names])

    lines = []
    for task, k in TASKS.items():
        yb = (y >= k).astype(int)
        score = p[:, k:].sum(1)
        thr = youden_threshold((vy[pros_val] >= k).astype(int), vp[pros_val][:, k:].sum(1))
        ops = {"argmax": (pred5 >= k).astype(int), f"val-thr={thr:.3f}": (score >= thr).astype(int)}
        print(f"\n===== {task} (test n={len(y)}, positives={yb.sum()}) =====")
        print(f"{'operating point':18} {'group':7} {'n':>4} {'pos':>4}  {'Accuracy':>22} {'Sensitivity':>22} {'Specificity':>22}  TP FN TN FP")
        r = REPORT[task]
        print(f"{'REPORT':18} {'all':7} {466:4} {'':>4}  {r['Accuracy']:22.4f} {r['Sensitivity']:22.4f} {r['Specificity']:22.4f}")
        for op, pred in ops.items():
            for grp in ["all", "JYE01", "Canon", "Nidek"]:
                mask = np.ones(len(y), bool) if grp == "all" else devices == grp
                if mask.sum() == 0:
                    continue
                m = binary_metrics(yb[mask], pred[mask])
                ci = bootstrap_ci(yb[mask], pred[mask], cli.n_boot)
                cells = [f"{m[s]:.4f} [{ci[s][0]:.3f},{ci[s][1]:.3f}]" for s in ("Accuracy", "Sensitivity", "Specificity")]
                print(f"{op:18} {grp:7} {mask.sum():4} {yb[mask].sum():4}  {cells[0]:>22} {cells[1]:>22} {cells[2]:>22}  "
                      f"{m['TP']:2} {m['FN']:2} {m['TN']:3} {m['FP']:2}")
                lines.append([task, op, grp, int(mask.sum()), int(yb[mask].sum()), m["Accuracy"], *ci["Accuracy"],
                              m["Sensitivity"], *ci["Sensitivity"], m["Specificity"], *ci["Specificity"],
                              m["TP"], m["FN"], m["TN"], m["FP"]])
            beat = all(binary_metrics(yb, pred)[s] >= r[s] for s in ("Sensitivity", "Specificity"))
            print(f"{'':18} -> beats report on Sens & Spec ({op}): {'YES' if beat else 'no'}")

    with open(os.path.join(out_dir, "mtm_vt_metrics.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "operating_point", "group", "n", "pos", "acc", "acc_lo", "acc_hi", "sens", "sens_lo", "sens_hi",
                    "spec", "spec_lo", "spec_hi", "TP", "FN", "TN", "FP"])
        w.writerows(lines)
    print(f"\nsaved predictions_*.csv and mtm_vt_metrics.csv to {out_dir}")


if __name__ == "__main__":
    main()
