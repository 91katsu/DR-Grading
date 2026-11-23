"""APTOS 2019 (Kaggle, 3662 labelled images) -> the same layout / preprocessing as Fundus_Dataset/DR5_512.

Input  (--src): train_1.csv / valid.csv / test.csv (id_code, diagnosis 0-4) and
                train_images/train_images, val_images/val_images, test_images/test_images (<id_code>.png)
Output (--dst): {train,val,test}/{0..4}/<id_code>.png  (crop retinal disc, pad square, resize, as preprocess_fundus.py)
                manifest.csv  (split, image, label, cohort=aptos, device=APTOS, patient=id_code, src)
                crop_boxes.csv

Example:
  python tools/prepare_aptos.py --src /mnt/10TB_2024/Katsu/data/DR/APTOS2019 --dst /mnt/10TB_2024/Katsu/data/DR/APTOS2019_512
"""
import argparse
import csv
import os
from collections import Counter
from multiprocessing import Pool

from preprocess_fundus import process  # tools/ is on sys.path when run as a script

SPLITS = {  # our split: (csv, image folder)
    "train": ("train_1.csv", os.path.join("train_images", "train_images")),
    "val": ("valid.csv", os.path.join("val_images", "val_images")),
    "test": ("test.csv", os.path.join("test_images", "test_images")),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    jobs, manifest = [], []
    for split, (csv_name, img_dir) in SPLITS.items():
        with open(os.path.join(args.src, csv_name), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                stem, label = r["id_code"], int(r["diagnosis"])
                assert 0 <= label <= 4, r
                src = os.path.join(args.src, img_dir, stem + ".png")
                if not os.path.exists(src):
                    raise FileNotFoundError(src)
                jobs.append((src, os.path.join(args.dst, split, str(label), stem + ".png"), args.size))
                manifest.append([split, stem + ".png", label, "aptos", "APTOS", stem, src])
    names = [m[1] for m in manifest]
    dup = [n for n, c in Counter(names).items() if c > 1]
    assert not dup, f"images in more than one split: {dup[:10]}"

    os.makedirs(args.dst, exist_ok=True)
    failed, rows = [], []
    with Pool(args.workers) as pool:
        for i, (src, info) in enumerate(pool.imap_unordered(process, jobs, chunksize=8), 1):
            if info is None:
                failed.append(src)
            else:
                rows.append([os.path.relpath(src, args.src), *info])
            if i % 500 == 0:
                print(f"{i}/{len(jobs)}", flush=True)

    with open(os.path.join(args.dst, "crop_boxes.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "orig_w", "orig_h", "x0", "y0", "x1", "y1"])
        w.writerows(sorted(rows))
    with open(os.path.join(args.dst, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "image", "label", "cohort", "device", "patient", "src"])
        w.writerows(manifest)

    for split in SPLITS:
        c = Counter(m[2] for m in manifest if m[0] == split)
        print(f"{split:5} {sum(c.values()):5}  grade 0-4 {[c[g] for g in range(5)]}")
    print(f"done: {len(rows)} ok, {len(failed)} failed", failed[:10])


if __name__ == "__main__":
    main()
