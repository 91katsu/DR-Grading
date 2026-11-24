"""Merge prepared datasets (same {train,val,test}/{0..4}/ layout + manifest.csv) into one --data_path,
with symlinks (no copies). The device column of manifest.csv is kept, so --train_devices / --val_devices
still select a source (e.g. --val_devices APTOS).

Example (stage 1 of the two-stage training on EyePACS + APTOS):
  python tools/merge_datasets.py --dst /mnt/10TB_2024/Katsu/data/DR/EyePACS_APTOS_512 \
      --src /mnt/10TB_2024/Katsu/data/DR/EyePACS_512 /mnt/10TB_2024/Katsu/data/DR/APTOS2019_512
"""
import argparse
import csv
import os
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--test_from_val", action="store_true",
                    help="no source has test rows (e.g. Fundus_FDR26_512, whose test/ = val): make <dst>/test = "
                         "links to the val images, since train_dr.py needs a test folder (NOT a test set)")
    args = ap.parse_args()

    rows, header, seen = [], None, {}
    for src in args.src:
        with open(os.path.join(src, "manifest.csv"), encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            header = header or r.fieldnames
            if r.fieldnames != header:
                raise ValueError(f"{src}/manifest.csv columns {r.fieldnames} != {header}")
            for m in r:
                if m["image"] in seen:
                    raise ValueError(f"image name {m['image']} in both {seen[m['image']]} and {src}")
                seen[m["image"]] = src
                link_src = os.path.join(src, m["split"], m["label"], m["image"])
                if not os.path.exists(link_src):
                    raise FileNotFoundError(link_src)
                rows.append((m, link_src))

    if os.path.exists(args.dst) and os.listdir(args.dst):
        raise SystemExit(f"{args.dst} exists and is not empty")
    for m, link_src in rows:
        d = os.path.join(args.dst, m["split"], m["label"])
        os.makedirs(d, exist_ok=True)
        os.symlink(link_src, os.path.join(d, m["image"]))
    if args.test_from_val:
        if any(m["split"] == "test" for m, _ in rows):
            raise SystemExit("--test_from_val: a source has test rows")
        for m, link_src in rows:
            if m["split"] == "val":
                d = os.path.join(args.dst, "test", m["label"])
                os.makedirs(d, exist_ok=True)
                os.symlink(link_src, os.path.join(d, m["image"]))
    with open(os.path.join(args.dst, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(m for m, _ in rows)

    for s in ("train", "val", "test"):
        c = Counter((m["device"], int(m["label"])) for m, _ in rows if m["split"] == s)
        devs = sorted({d for d, _ in c})
        print(f"{s:5} " + " | ".join(f"{d} {sum(c[d, g] for g in range(5))} {[c[d, g] for g in range(5)]}"
                                    for d in devs))
    print(f"done: {len(rows)} links in {args.dst}")


if __name__ == "__main__":
    main()
