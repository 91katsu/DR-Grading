"""Build a de-duplicated copy of an ImageFolder DR dataset from the duplicate lists in problem*.txt.

problem.txt  : groups of byte-identical images (MD5) saved under different names (JY11301 retrospective)
problem2.txt : near-duplicate pairs (A: same photo, B: very similar, needs visual check)
Groups that share an image (e.g. 00073-1234.png in problem2 groups 14 and 15) are merged.

Rule per merged group (train split only; val/test are copied unchanged):
  * all copies have the same DR grade -> keep the first train copy, drop the other train copies
  * DR grades disagree                -> drop every train copy (label is unreliable)

Files are hard links (no extra disk space) when src and dst are on the same drive, else copies.
Writes <dst>/manifest.csv (images present in the new dataset) and <dst>/removed_duplicates.csv.

Example:
  python tools/dedup_dataset.py --src Fundus_Dataset/DR5_512_prosval --dst Fundus_Dataset/DR5_512_prosval_dedup \
      --problems D:/data/fundus_yolo/problem.txt D:/data/fundus_yolo/problem2.txt
"""
import argparse
import csv
import os
import re
import shutil
from collections import Counter

IMG_RE = re.compile(r"^\s+(?:train|test\(val\))\s+(\S+\.png)", re.M)


def read_groups(paths, extra=()):
    """extra: additional (group_id, [images]) groups, e.g. from an MD5 scan."""
    groups = list(extra)
    for path in paths:
        txt = open(path, encoding="utf-8-sig").read()
        for block in re.split(r"\n\[第 ", txt)[1:]:
            gid = f"{os.path.basename(path)}#{block.split(']')[0].strip()}"
            imgs = list(dict.fromkeys(IMG_RE.findall(block)))
            if len(imgs) >= 2:
                groups.append((gid, imgs))
    # merge groups sharing an image (union-find)
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, imgs in groups:
        for i in imgs[1:]:
            parent[find(i)] = find(imgs[0])
    merged = {}
    for gid, imgs in groups:
        r = find(imgs[0])
        ids, members = merged.setdefault(r, ([], []))
        ids.append(gid)
        members.extend(i for i in imgs if i not in members)
    return list(merged.values())


def place(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--problems", nargs="+", required=True)
    cli = ap.parse_args()
    if os.path.exists(cli.dst):
        raise SystemExit(f"{cli.dst} already exists; remove it first")

    with open(os.path.join(cli.src, "manifest.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())
    # images actually present in src (val of *_prosval is a subset of the manifest's val)
    present = {}
    for split in ("train", "val", "test"):
        for cls in sorted(os.listdir(os.path.join(cli.src, split))):
            for f in os.listdir(os.path.join(cli.src, split, cls)):
                present[(split, f)] = os.path.join(cli.src, split, cls, f)
    label_of = {r["image"]: r["label"] for r in rows}  # all splits, incl. unused retro val copies

    removed = []
    for gids, imgs in read_groups(cli.problems):
        missing = [i for i in imgs if i not in label_of]
        if missing:
            raise SystemExit(f"{missing} (groups {gids}) not in manifest")
        train_imgs = [i for i in imgs if ("train", i) in present]
        grades = {label_of[i] for i in imgs}
        group_desc = f"{'+'.join(gids)}: " + ", ".join(f"{i}(DR{label_of[i]})" for i in imgs)
        if len(grades) > 1:
            removed += [(i, label_of[i], "grades_disagree", group_desc) for i in train_imgs]
        else:
            removed += [(i, label_of[i], "duplicate_same_grade", group_desc) for i in train_imgs[1:]]
    drop = {i for i, *_ in removed}

    kept = Counter()
    for (split, f), path in present.items():
        if split == "train" and f in drop:
            continue
        cls = os.path.basename(os.path.dirname(path))
        place(path, os.path.join(cli.dst, split, cls, f))
        kept[(split, cls)] += 1

    keep_names = {(s, f) for (s, f) in present if not (s == "train" and f in drop)}
    with open(os.path.join(cli.dst, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(r for r in rows if (r["split"], r["image"]) in keep_names)
    with open(os.path.join(cli.dst, "removed_duplicates.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "label", "reason", "group"])
        w.writerows(sorted(removed))

    print(f"removed {len(drop)} train images: {dict(Counter(r[2] for r in removed))}")
    print(f"removed per grade: {dict(sorted(Counter(r[1] for r in removed).items()))}")
    for split in ("train", "val", "test"):
        per = [kept[(split, str(c))] for c in range(5)]
        n0 = sum(1 for s, _ in present if s == split)
        print(f"{split}: {n0} -> {sum(per)}  per grade {per}")
    print(f"wrote {cli.dst}")


if __name__ == "__main__":
    main()
