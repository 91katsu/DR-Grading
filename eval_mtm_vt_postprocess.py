"""mtmDR / vtDR evaluation of classifier predictions combined with saved detections through the
deployed DR post-processing (ALFA), compared with the original paper.  CPU only, no torch.

Inputs
  --pred_csv  predictions_{val,test}.csv written by eval_mtm_vt.py
              (columns image, cohort, device, label, pred, p0..p4)  ->  Cls = argmax(p), cls_prob = max(p)
  --det_root  raw detections, one json per image: <det_root>/<device>/raw/<image stem>.json
              {boxes (x1 y1 x2 y2, pixels of the cropped + padded image), labels, scores, offset [dx, dy]}
              e.g. D:/exp_result/fundus/Det_Raw/det_raw_0JCN (all boxes of the detection service,
              no NMS / threshold / reconciliation applied)

Per image, exactly as the deployed service (postprocess/ = unmodified copy of postprocess v2.1.9):
  postprocess.postprocess(): IoM-NMS 0.5 -> box + offset (original image) -> ALFA.contradict_DR5 -> lesion thresholds
  lesion thresholds : --threshold_config (default postprocess/threshold_config.json = deployed),
                      type 0 by age group (--age, API default 40), type 1 single set
  ALFA tables       : type 0 -> cls_prob_jye01.csv / Score_jye01.csv, type 1 -> cls_prob.csv / Score.csv
  type per device   : --type_map (default: type 0 for all devices)
Not reproduced: the API quality filter and spot-light elimination (their outputs are not saved).

Outputs: always printed to the terminal; files only with --out_dir (nothing is written without it)
  predictions_postprocess.csv   image, cohort, device, label, cls_raw, cls_prob, det_init, cls_alfa, changed,
                                n_boxes_nms, n_boxes_final   (det_init = grade of the NMS boxes before ALFA)
  mtm_vt_metrics_postprocess.csv  same columns as mtm_vt_metrics.csv, method = raw (argmax) / alfa
  run_config.json               arguments, lesion thresholds per type, sha256 of the postprocess/ files
  boxes_postprocess.json        (--save_boxes, needs --image_root and --out_dir) final boxes, normalised cx cy w h, as the API

Example
  python eval_mtm_vt_postprocess.py --pred_csv output_dir/DR5_swinMS_desktop_v1/predictions_val.csv \\
      --det_root D:/exp_result/fundus/Det_Raw/det_raw_0JCN --age 40
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from postprocess.postprocess import postprocess  # noqa: E402  deployed code, unmodified
from postprocess.nms import iom_nms  # noqa: E402
from postprocess.ALFA import det2cls  # noqa: E402
from utils.metrics import REPORT, TASKS, binary_metrics, bootstrap_ci, wald_ci  # noqa: E402

DEVICES = ["JYE01", "Canon", "Nidek"]
IMG_EXTS = [".jpg", ".JPG", ".jpeg", ".png", ".PNG", ".tif", ".bmp"]
DEFAULT_THRESHOLD_CONFIG = REPO / "postprocess" / "threshold_config.json"  # = deployed v2.1.9
PLACEHOLDER_SHAPE = (1, 1, 3)  # image size only changes the normalised output boxes, not the DR grade


def thresholds_for(type_id, age, config):
    """Same as deployment postprocess/main.py get_score_threshold_for_single_age + match_age_rule."""
    tc = config["score_threshold_by_type"][str(type_id)]
    if "age_rules" in tc:
        for rule in tc["age_rules"]:
            lo, hi = rule.get("min_age"), rule.get("max_age")
            if (lo is None or age >= lo) and (hi is None or age < hi):
                thr = rule["thresholds"]
                break
        else:
            raise ValueError(f"no age rule matched for age={age}")
    else:
        thr = tc["thresholds"]
    return {int(k): float(v) for k, v in thr.items()}


def parse_type_map(s):
    out = {}
    for item in s.split(","):
        dev, t = item.split("=")
        if t.strip() not in ("0", "1"):
            raise ValueError(f"--type_map: type must be 0 or 1, got {item}")
        out[dev.strip()] = int(t)
    return out


def image_shape(image_root, device, stem):
    from PIL import Image
    for e in IMG_EXTS:
        p = Path(image_root) / device / "images" / f"{stem}{e}"
        if p.exists():
            with Image.open(p) as im:
                w, h = im.size
            return (h, w, 3)
    raise FileNotFoundError(f"original image not found: {image_root}/{device}/images/{stem}.*")


def quadratic_kappa(y, p, k=5):
    o = np.zeros((k, k))
    for a, b in zip(y, p):
        o[a, b] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return 1 - (w * o).sum() / (w * e).sum()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_csv", required=True, help="predictions_val.csv / predictions_test.csv from eval_mtm_vt.py")
    ap.add_argument("--det_root", required=True, help="<det_root>/<device>/raw/<stem>.json")
    ap.add_argument("--age", type=int, default=40, help="age for the type-0 lesion thresholds (API default 40)")
    ap.add_argument("--threshold_config", default=None,
                    help="lesion threshold json (same format as postprocess/threshold_config.json); "
                         "omitted = postprocess/threshold_config.json (deployed), "
                         "e.g. postprocess/threshold_config_tuned.json")
    ap.add_argument("--type_map", default="JYE01=0,Canon=0,Nidek=0",
                    help="postprocess type per device (0 = JYE01 tables + age thresholds, 1 = desktop)")
    ap.add_argument("--image_root", default=None,
                    help="original images <image_root>/<device>/images/<stem>.*, only for the real image size "
                         "(needed by --save_boxes; the DR grade does not depend on it)")
    ap.add_argument("--save_boxes", action="store_true", help="write boxes_postprocess.json (needs --image_root)")
    ap.add_argument("--ci", default="wald", choices=["wald", "bootstrap"],
                    help="95%% CI: wald (as the original report) or bootstrap (as eval_mtm_vt.py)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out_dir", default=None, help="write the result files here; omitted = print to the terminal only")
    cli = ap.parse_args()
    if cli.save_boxes and not (cli.image_root and cli.out_dir):
        ap.error("--save_boxes needs --image_root and --out_dir")

    pred_csv = Path(cli.pred_csv).resolve()
    det_root = Path(cli.det_root).resolve()
    image_root = Path(cli.image_root).resolve() if cli.image_root else None
    out_dir = Path(cli.out_dir).resolve() if cli.out_dir else None  # None: print only, write nothing
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    type_map = parse_type_map(cli.type_map)
    thr_config = Path(cli.threshold_config).resolve() if cli.threshold_config else DEFAULT_THRESHOLD_CONFIG
    config = json.loads(thr_config.read_text(encoding="utf-8"))
    thr_by_type = {t: thresholds_for(t, cli.age, config) for t in set(type_map.values())}

    with open(pred_csv, encoding="utf-8-sig") as fh:
        preds = list(csv.DictReader(fh))
    missing = [r["image"] for r in preds if r["device"] not in type_map
               or not (det_root / r["device"] / "raw" / f"{Path(r['image']).stem}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images without detection json or --type_map entry, e.g. {missing[:5]}")

    os.chdir(REPO)  # postprocess() opens its tables as ./postprocess/<table>.csv
    rows, boxes_out = [], {}
    for r in preds:
        stem, dev = Path(r["image"]).stem, r["device"]
        t = type_map[dev]
        thr = thr_by_type[t]
        probs = [float(r[f"p{i}"]) for i in range(5)]
        c_raw, c_prob = int(np.argmax(probs)), float(np.max(probs))
        j = json.loads((det_root / dev / "raw" / f"{stem}.json").read_text(encoding="utf-8"))
        dx, dy = j["offset"]
        shape = image_shape(image_root, dev, stem) if image_root else PLACEHOLDER_SHAPE
        b, s, l = np.array(j["boxes"]).reshape(-1, 4), np.array(j["scores"]), np.array(j["labels"])

        keep = iom_nms(b, s, iom_threshold=0.5) if len(s) else []
        det_init, _ = det2cls({"boxes": b[keep] + [dx, dy, dx, dy], "scores": s[keep], "labels": l[keep]}, thr, shape)

        with contextlib.redirect_stdout(io.StringIO()):
            fb, fs, fl, c_alfa = postprocess(j["boxes"], j["scores"], j["labels"], thr,
                                             {"left": dx, "top": dy}, {"pad_left": 0, "pad_top": 0},
                                             c_raw, c_prob, shape, type=t)
        rows.append({"image": r["image"], "cohort": r.get("cohort", ""), "device": dev, "label": int(r["label"]),
                     "cls_raw": c_raw, "cls_prob": round(c_prob, 5), "det_init": int(det_init),
                     "cls_alfa": int(c_alfa), "changed": int(c_raw != c_alfa),
                     "n_boxes_nms": len(keep), "n_boxes_final": len(fl)})
        if cli.save_boxes:
            boxes_out[r["image"]] = {"boxes": np.asarray(fb).tolist(), "scores": np.asarray(fs).tolist(),
                                     "labels": np.asarray(fl).astype(int).tolist(), "cls": int(c_alfa)}

    # ---------------- metrics ----------------
    y = np.array([r["label"] for r in rows])
    devices = np.array([r["device"] for r in rows])
    methods = {"raw": np.array([r["cls_raw"] for r in rows]), "alfa": np.array([r["cls_alfa"] for r in rows])}
    print(f"{pred_csv}\n+ {det_root}  | age {cli.age} | type {cli.type_map} | CI {cli.ci} | n = {len(rows)}")
    print(f"lesion thresholds: {thr_config}" + ("  (default)" if thr_config == DEFAULT_THRESHOLD_CONFIG else ""))
    for t, v in sorted(thr_by_type.items()):
        print(f"  type {t}: " + " ".join(f"{k}:{v[k]:g}" for k in sorted(v)))

    for name, p in methods.items():
        cm = np.zeros((5, 5), int)
        for a, b in zip(y, p):
            cm[a, b] += 1
        print(f"\n[{name} 5-class] acc {np.mean(p == y):.4f} | quadratic kappa {quadratic_kappa(y, p):.4f}")
        print("confusion matrix (rows=GT 0-4, cols=pred 0-4):\n", cm)

    raw, alfa = methods["raw"], methods["alfa"]
    ch = raw != alfa
    print(f"\nALFA changed {ch.sum()} images (up {(alfa > raw).sum()}, down {(alfa < raw).sum()}): "
          f"fixed {(ch & (alfa == y)).sum()}, broke {(ch & (raw == y)).sum()}")

    lines = []
    for task, k in TASKS.items():
        yb = (y >= k).astype(int)
        print(f"\n===== {task} (n={len(y)}, positives={yb.sum()}) =====")
        print(f"{'method':7} {'group':7} {'n':>4} {'pos':>4}  {'Accuracy':>24} {'Sensitivity':>24} {'Specificity':>24}  TP FN  TN FP")
        r = REPORT[task]
        print(f"{'REPORT':7} {'all':7} {466:4} {'':>4}  {r['Accuracy']:24.4f} {r['Sensitivity']:24.4f} {r['Specificity']:24.4f}")
        for grp in ["all"] + DEVICES:  # grouped by device (all first), raw / alfa next to each other
            mask = np.ones(len(y), bool) if grp == "all" else devices == grp
            if mask.sum() == 0:
                continue
            for name, p in methods.items():
                pred = (p >= k).astype(int)
                m = binary_metrics(yb[mask], pred[mask])
                ci = wald_ci(m) if cli.ci == "wald" else bootstrap_ci(yb[mask], pred[mask], cli.n_boot)
                cells = [f"{m[s]:.4f} [{ci[s][0]:.4f}, {ci[s][1]:.4f}]" for s in ("Accuracy", "Sensitivity", "Specificity")]
                print(f"{name:7} {grp:7} {mask.sum():4} {yb[mask].sum():4}  {cells[0]:>24} {cells[1]:>24} {cells[2]:>24}  "
                      f"{m['TP']:2} {m['FN']:2} {m['TN']:3} {m['FP']:2}")
                lines.append([task, name, grp, int(mask.sum()), int(yb[mask].sum()), m["Accuracy"], *ci["Accuracy"],
                              m["Sensitivity"], *ci["Sensitivity"], m["Specificity"], *ci["Specificity"],
                              m["TP"], m["FN"], m["TN"], m["FP"]])

    if out_dir is None:
        return

    with open(out_dir / "predictions_postprocess.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if cli.save_boxes:
        (out_dir / "boxes_postprocess.json").write_text(json.dumps(boxes_out), encoding="utf-8")

    with open(out_dir / "mtm_vt_metrics_postprocess.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "method", "group", "n", "pos", "acc", "acc_lo", "acc_hi", "sens", "sens_lo", "sens_hi",
                    "spec", "spec_lo", "spec_hi", "TP", "FN", "TN", "FP"])
        w.writerows(lines)

    pp = REPO / "postprocess"
    run_cfg = {"pred_csv": str(pred_csv), "det_root": str(det_root), "age": cli.age, "type_map": type_map,
               "image_root": str(image_root) if image_root else None, "ci": cli.ci,
               "threshold_config": str(thr_config), "threshold_config_sha256": sha256(thr_config),
               "lesion_thresholds": {str(t): v for t, v in thr_by_type.items()},
               "postprocess_sha256": {f.name: sha256(f) for f in sorted(pp.iterdir()) if f.suffix in (".py", ".csv", ".json")}}
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved predictions_postprocess.csv, mtm_vt_metrics_postprocess.csv, run_config.json to {out_dir}")


if __name__ == "__main__":
    main()
