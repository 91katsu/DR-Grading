"""EyePACS (Kaggle 2015 DR detection, train part: 35126 labelled images, 17563 patients x left/right)
-> the same layout / preprocessing as Fundus_Dataset/DR5_512 (crop retinal disc, pad square, resize).

Input  (--src): trainLabels.csv (image, level 0-4) and train/<image>.jpeg
Output (--dst): {train,val,test}/{0..4}/<image>.png
                manifest.csv  (split, image, label, cohort=eyepacs, device=EyePACS, patient, src)
                crop_boxes.csv (image, orig_w, orig_h, x0, y0, x1, y1)
Split: by PATIENT (both eyes of a person in the same split), stratified by the patient's worse eye,
       --val_frac / --test_frac of the patients (default 5% / 5%), fixed --seed.

Example:
  python tools/prepare_eyepacs.py --src /mnt/10TB_2024/Katsu/data/DR/EyePACS/extracted \
      --dst /mnt/10TB_2024/Katsu/data/DR/EyePACS_512
"""
import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from multiprocessing import Pool

from preprocess_fundus import process  # tools/ is on sys.path when run as a script


def split_patients(levels, val_frac, test_frac, seed):
    """levels: patient -> list of levels. Returns patient -> split, stratified by max level."""
    by_grade = defaultdict(list)
    for p, lv in sorted(levels.items()):
        by_grade[max(lv)].append(p)
    rng = random.Random(seed)
    split = {}
    for g in sorted(by_grade):
        ps = by_grade[g]
        rng.shuffle(ps)
        n_val, n_test = round(len(ps) * val_frac), round(len(ps) * test_frac)
        for i, p in enumerate(ps):
            split[p] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    with open(os.path.join(args.src, "trainLabels.csv"), encoding="utf-8") as fh:
        labels = {r["image"]: int(r["level"]) for r in csv.DictReader(fh)}
    assert all(0 <= v <= 4 for v in labels.values())
    patient = {n: n.rsplit("_", 1)[0] for n in labels}  # "10_left" -> "10"
    levels = defaultdict(list)
    for n, lv in labels.items():
        levels[patient[n]].append(lv)
    split = split_patients(levels, args.val_frac, args.test_frac, args.seed)

    jobs, manifest = [], {}
    for n, lv in sorted(labels.items()):
        src = os.path.join(args.src, "train", n + ".jpeg")
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        s = split[patient[n]]
        jobs.append((src, os.path.join(args.dst, s, str(lv), n + ".png"), args.size))
        manifest[src] = [s, n + ".png", lv, "eyepacs", "EyePACS", patient[n], src]

    os.makedirs(args.dst, exist_ok=True)
    failed, boxes, odd = [], [], []
    with Pool(args.workers) as pool:
        for i, (src, info) in enumerate(pool.imap_unordered(process, jobs, chunksize=16), 1):
            if info is None:
                failed.append(src)
            else:
                w, h, x0, y0, x1, y1 = info
                boxes.append([os.path.basename(src), *info])
                # disc box = whole image (no dark border found) or very small box: check these by eye
                if (x1 - x0) * (y1 - y0) >= 0.99 * w * h or (x1 - x0) < 0.3 * w:
                    odd.append(os.path.basename(src))
            if i % 2000 == 0:
                print(f"{i}/{len(jobs)}", flush=True)

    ok = [m for src, m in manifest.items() if src not in set(failed)]
    with open(os.path.join(args.dst, "crop_boxes.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "orig_w", "orig_h", "x0", "y0", "x1", "y1"])
        w.writerows(sorted(boxes))
    with open(os.path.join(args.dst, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "image", "label", "cohort", "device", "patient", "src"])
        w.writerows(ok)
    with open(os.path.join(args.dst, "odd_crops.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(odd)) + "\n")

    for s in ("train", "val", "test"):
        c = Counter(m[2] for m in ok if m[0] == s)
        n_pat = len({m[5] for m in ok if m[0] == s})
        print(f"{s:5} {sum(c.values()):6} images {n_pat:6} patients  grade 0-4 {[c[g] for g in range(5)]}")
    print(f"done: {len(ok)} ok, {len(failed)} failed {failed[:10]}; "
          f"{len(odd)} odd crops (whole image or narrow box) listed in odd_crops.txt")


if __name__ == "__main__":
    main()
