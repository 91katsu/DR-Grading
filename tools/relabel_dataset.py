"""Copy a prepared dataset (e.g. DR5_512) with new labels for some images, e.g. the 3-doctor labels of the
prospective images (DR_label_merged.xlsx). Same split, same images; only the class folder of the relabelled
images changes. Images that are not in the label file (retrospective) keep their old label.

Label file (.xlsx needs openpyxl, or .csv): columns
  filename, label, label_source, split_valpros_dedup (train / val / "not used")
Checks: every label-file image with a split is in the source manifest, with the same split; "not used" images
are not in it.

Output (--dst): {train,val,test}/{0..4}/<image>.png (copies), removed_duplicates.csv (copied),
  manifest.csv      = source manifest + label_orig (old label) + label_source ("original" when not relabelled)
  label_changes.csv = images whose label changed (image, split, device, label_orig, label, label_source)

Example (base conda env, which has pandas + openpyxl):
  python tools/relabel_dataset.py --src /mnt/10TB_2024/Katsu/data/DR/DR5_512 \
      --labels /mnt/10TB_2024/Katsu/data/DR/DR_Label/DR_label_merged.xlsx \
      --dst /mnt/10TB_2024/Katsu/data/DR/DR5_512_consensus
  --drop_source "chen (no consensus)" leaves out the images without a doctors' consensus.
"""
import argparse
import csv
import os
import shutil
from collections import Counter

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--drop_source", nargs="*", default=[], help="leave out images with this label_source")
    cli = ap.parse_args()

    read = pd.read_csv if cli.labels.endswith(".csv") else pd.read_excel
    lab = read(cli.labels)
    if lab.filename.duplicated().any():
        raise SystemExit(f"duplicate filenames in {cli.labels}")
    lab = {r.filename: r for r in lab.itertuples(index=False)}

    with open(os.path.join(cli.src, "manifest.csv"), encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header, rows = reader.fieldnames, list(reader)
    names = {r["image"]: r for r in rows}
    errors = []
    for n, r in lab.items():
        if r.split_valpros_dedup == "not used":
            if n in names:
                errors.append(f"{n}: 'not used' in the label file but in the manifest ({names[n]['split']})")
        elif n not in names:
            errors.append(f"{n}: not in the manifest")
        elif names[n]["split"] != r.split_valpros_dedup:
            errors.append(f"{n}: split {names[n]['split']} (manifest) vs {r.split_valpros_dedup} (label file)")
    if errors:
        raise SystemExit("label file does not fit the dataset:\n  " + "\n  ".join(errors[:20]))

    if os.path.exists(cli.dst) and os.listdir(cli.dst):
        raise SystemExit(f"{cli.dst} exists and is not empty")
    out, changes, dropped = [], [], Counter()
    for m in rows:
        r = lab.get(m["image"])
        new = dict(m, label_orig=m["label"], label_source="original")
        if r is not None:
            new["label"], new["label_source"] = str(int(r.label)), r.label_source
        if new["label_source"] in cli.drop_source:
            dropped[m["split"]] += 1
            continue
        src = os.path.join(cli.src, m["split"], m["label"], m["image"])
        d = os.path.join(cli.dst, m["split"], new["label"])
        os.makedirs(d, exist_ok=True)
        shutil.copy2(src, os.path.join(d, m["image"]))
        out.append(new)
        if new["label"] != m["label"]:
            changes.append({k: new[k] for k in ("image", "split", "device", "label_orig", "label", "label_source")})

    for f in ("removed_duplicates.csv",):
        if os.path.exists(os.path.join(cli.src, f)):
            shutil.copy2(os.path.join(cli.src, f), cli.dst)
    with open(os.path.join(cli.dst, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header + ["label_orig", "label_source"])
        w.writeheader()
        w.writerows(out)
    with open(os.path.join(cli.dst, "label_changes.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "split", "device", "label_orig", "label", "label_source"])
        w.writeheader()
        w.writerows(changes)

    for s in ("train", "val", "test"):
        part = [m for m in out if m["split"] == s]
        c = Counter(int(m["label"]) for m in part)
        ch = [x for x in changes if x["split"] == s]
        mtm = sum((int(x["label_orig"]) >= 2) != (int(x["label"]) >= 2) for x in ch)
        print(f"{s:5} {len(part):5} images, grade 0-4 {[c[g] for g in range(5)]}, "
              f"{len(ch)} relabelled ({mtm} cross mtmDR), {dropped[s]} dropped")
    print(f"done: {len(out)} images in {cli.dst}, sources {dict(Counter(m['label_source'] for m in out))}")


if __name__ == "__main__":
    main()
