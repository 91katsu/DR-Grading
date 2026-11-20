"""Build the split: train = all retro train + all prospective train (minus duplicates),
val = prospective test (466, used for checkpoint selection), test = retrospective test (1181).

Source: an ImageFolder dataset made by prepare_dr_data.py (default Fundus_Dataset/DR5_512), whose manifest has
  train/retro 4711, train/pros 1554, val/retro 1181, val/pros 281, test/pros 466.
Mapping: new train = train + val/pros ; new val = test (pros) ; new test = val/retro.

Duplicates (problem*.txt + an MD5 scan of all files, groups sharing an image merged; see dedup_dataset.py) are removed from TRAIN only,
so val/test keep their original sizes:
  * a copy of the group is in val/test           -> drop every train copy (no train/test leakage)
  * otherwise, all copies have the same DR grade  -> keep the first train copy
  * otherwise (grades disagree)                   -> drop every train copy

Example:
  python tools/build_valpros_dataset.py --src Fundus_Dataset/DR5_512 --dst Fundus_Dataset/DR5_512_valpros_dedup \
      --problems D:/data/fundus_yolo/problem.txt D:/data/fundus_yolo/problem2.txt
"""
import argparse
import csv
import hashlib
import os
from collections import Counter, defaultdict

from dedup_dataset import place, read_groups  # tools/ is on sys.path when run as a script


def new_split(row):
    if row["split"] == "train" or (row["split"] == "val" and row["cohort"] == "pros"):
        return "train"
    if row["split"] == "test":
        return "val"
    return "test"  # val/retro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Fundus_Dataset/DR5_512")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--problems", nargs="+", required=True)
    cli = ap.parse_args()
    if os.path.exists(cli.dst):
        raise SystemExit(f"{cli.dst} already exists; remove it first")

    with open(os.path.join(cli.src, "manifest.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())
    old = {r["image"]: r for r in rows}
    if len(old) != len(rows):
        raise SystemExit("duplicate image names in manifest")
    split_of = {n: new_split(r) for n, r in old.items()}
    label_of = {n: r["label"] for n, r in old.items()}

    # byte-identical files anywhere in the dataset (e.g. prospective 047NJ-* / 048NJ-*) are added as extra groups
    by_md5 = defaultdict(list)
    for n, r in old.items():
        with open(os.path.join(cli.src, r["split"], r["label"], n), "rb") as fh:
            by_md5[hashlib.md5(fh.read()).hexdigest()].append(n)
    extra = [(f"md5:{h[:8]}", sorted(v)) for h, v in by_md5.items() if len(v) > 1]
    print(f"MD5 scan: {len(extra)} groups of byte-identical files")

    removed = []
    for gids, imgs in read_groups(cli.problems, extra):
        missing = [i for i in imgs if i not in old]
        if missing:
            raise SystemExit(f"{missing} (groups {gids}) not in manifest")
        train_imgs = [i for i in imgs if split_of[i] == "train"]
        desc = f"{'+'.join(gids)}: " + ", ".join(f"{i}({split_of[i]},DR{label_of[i]})" for i in imgs)
        if any(split_of[i] != "train" for i in imgs):
            removed += [(i, label_of[i], "copy_in_val_or_test", desc) for i in train_imgs]
        elif len({label_of[i] for i in imgs}) > 1:
            removed += [(i, label_of[i], "grades_disagree", desc) for i in train_imgs]
        else:
            removed += [(i, label_of[i], "duplicate_same_grade", desc) for i in train_imgs[1:]]
    drop = {i for i, *_ in removed}

    out_rows, kept = [], Counter()
    for n, r in old.items():
        if n in drop:
            continue
        src = os.path.join(cli.src, r["split"], r["label"], n)
        if not os.path.isfile(src):
            raise SystemExit(f"missing file {src}")
        s = split_of[n]
        place(src, os.path.join(cli.dst, s, r["label"], n))
        out_rows.append({**r, "split": s})
        kept[(s, r["cohort"], r["label"])] += 1

    with open(os.path.join(cli.dst, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(out_rows)
    with open(os.path.join(cli.dst, "removed_duplicates.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "label", "reason", "group"])
        w.writerows(sorted(removed))

    print(f"removed {len(drop)} train images: {dict(Counter(r[2] for r in removed))}")
    print(f"removed per grade: {dict(sorted(Counter(r[1] for r in removed).items()))}")
    for s in ("train", "val", "test"):
        for coh in ("retro", "pros"):
            per = [kept[(s, coh, str(c))] for c in range(5)]
            if sum(per):
                print(f"{s:5} {coh:5} {sum(per):5}  per grade {per}")
    print(f"wrote {cli.dst}")


if __name__ == "__main__":
    main()
