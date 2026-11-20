"""Build an ImageFolder tree for 5-class DR grading (ICDR 0-4) from the JY11301/JY11302 handover data.

Layout produced:
    <out>/{train,val,test}/{0,1,2,3,4}/<image>.png
    <out>/manifest.csv   (split, label, cohort, device, patient, src)

Split definition (ground truth = Dr. Chen's labels, taken from the split files):
    train = retrospective chen/train.txt + prospective train (JYE01 chen/train.txt,
            Canon chen/set/train_Canon.txt, Nidek chen/set/train_Nedik.txt)
            minus a patient-level, grade-stratified 15% of the prospective train -> val
    val   = that prospective hold-out + retrospective chen/test.txt
    test  = prospective JYE01 chen/test.txt + set/test_Canon.txt + set/test_Nedik.txt
            (466 images, 90 mtmDR+, 22 vtDR+; the original paper's test set)
"""
import argparse
import collections
import csv
import os
import random
import shutil

DEFAULT_SRC = r"D:\原始交接檔案\眼底AI交接\Dataset"
RETRO = "回溯型(JY11301)"
PROS = "前瞻型(JY11302)"
DEVICE = {"1": "JYE01", "2": "Canon", "3": "Nidek"}


def read_split(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if parts:
                rows.append((parts[0], int(parts[1])))
    return rows


def read_labels_csv(path):
    labels = {}
    with open(path, encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            dr = next((x for x in row[1:] if x.startswith("DR ")), None)
            if row and dr and dr[3:].isdigit():
                labels[row[0] + ".png"] = int(dr[3:])
    return labels


def meta(name, cohort):
    stem = name[:-4]
    if cohort == "retro":
        return stem.split("-")[0], "Canon"
    pt, code = stem.split("-")
    return pt[:3], DEVICE[code[0]]


def holdout_patients(rows, frac, seed):
    """Pick ~frac of patients per worst-grade stratum (patient-level, so val is not leaked from train)."""
    worst = collections.defaultdict(int)
    for name, label in rows:
        pt, _ = meta(name, "pros")
        worst[pt] = max(worst[pt], label)
    strata = collections.defaultdict(list)
    for pt, g in sorted(worst.items()):
        strata[g].append(pt)
    rng = random.Random(seed)
    held = set()
    for g in sorted(strata):
        pts = strata[g]
        rng.shuffle(pts)
        held.update(pts[: max(1, round(len(pts) * frac))])
    return held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Fundus_Dataset", "DR5"))
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    r_lab = os.path.join(args.src, RETRO, "label", "chen")
    p_lab = os.path.join(args.src, PROS, "label", "chen")
    img_dir = {"retro": os.path.join(args.src, RETRO, "image"), "pros": os.path.join(args.src, PROS, "image")}

    pros_train_files = [os.path.join(p_lab, "train.txt"),
                        os.path.join(p_lab, "set", "train_Canon.txt"),
                        os.path.join(p_lab, "set", "train_Nedik.txt")]
    pros_test_files = [os.path.join(p_lab, "test.txt"),
                       os.path.join(p_lab, "set", "test_Canon.txt"),
                       os.path.join(p_lab, "set", "test_Nedik.txt")]

    records = []  # (split, name, label, cohort)
    for name, label in read_split(os.path.join(r_lab, "train.txt")):
        records.append(("train", name, label, "retro"))
    for name, label in read_split(os.path.join(r_lab, "test.txt")):
        records.append(("val", name, label, "retro"))
    # patients are imaged on 2 devices, so choose held-out patients across all device files at once
    pros_train = [row for f in pros_train_files for row in read_split(f)]
    held = holdout_patients(pros_train, args.val_frac, args.seed)
    for name, label in pros_train:
        split = "val" if meta(name, "pros")[0] in held else "train"
        records.append((split, name, label, "pros"))
    for f in pros_test_files:
        for name, label in read_split(f):
            records.append(("test", name, label, "pros"))

    # ---- sanity checks ----
    names = [r[1] for r in records]
    dup = [n for n, c in collections.Counter(names).items() if c > 1]
    assert not dup, f"images in more than one split: {dup[:10]}"
    for split, name, label, cohort in records:
        assert 0 <= label <= 4, (name, label)
        assert os.path.exists(os.path.join(img_dir[cohort], name)), f"missing image {name}"
    gt = {"retro": read_labels_csv(os.path.join(r_lab, "labels.csv")),
          "pros": read_labels_csv(os.path.join(p_lab, "labels.csv"))}
    mismatch = [(n, l, gt[c].get(n)) for _, n, l, c in records if gt[c].get(n) != l]
    print(f"split-file label vs chen labels.csv mismatches: {len(mismatch)}", mismatch[:10])

    # ---- summary ----
    print(f"\n{'split':6} {'cohort':6} {'device':6} {'n':>5}  grade 0/1/2/3/4        mtm+  vt+")
    groups = collections.defaultdict(list)
    for split, name, label, cohort in records:
        groups[(split, cohort, meta(name, cohort)[1])].append(label)
        groups[(split, "ALL", "")].append(label)
    for key in sorted(groups, key=lambda k: (["train", "val", "test"].index(k[0]), k[1] == "ALL", k[1], k[2])):
        labs = groups[key]
        c = collections.Counter(labs)
        print(f"{key[0]:6} {key[1]:6} {key[2]:6} {len(labs):5}  "
              f"{'/'.join(str(c[g]) for g in range(5)):22} {sum(l >= 2 for l in labs):5} {sum(l >= 3 for l in labs):4}")

    val_pts = {meta(n, c)[0] for s, n, _, c in records if s == "val" and c == "pros"}
    train_pts = {meta(n, c)[0] for s, n, _, c in records if s == "train" and c == "pros"}
    print(f"\nprospective patients: train {len(train_pts)}, val {len(val_pts)}, overlap {len(train_pts & val_pts)}")
    assert not (train_pts & val_pts), "prospective val patients leak into train"

    test = [r for r in records if r[0] == "test"]
    assert (len(test), sum(r[2] >= 2 for r in test), sum(r[2] >= 3 for r in test)) == (466, 90, 22), "test set != report"

    if args.dry_run:
        return

    # ---- copy ----
    for split in ("train", "val", "test"):
        for g in range(5):
            os.makedirs(os.path.join(args.out, split, str(g)), exist_ok=True)
    copied = 0
    for split, name, label, cohort in records:
        dst = os.path.join(args.out, split, str(label), name)
        src = os.path.join(img_dir[cohort], name)
        if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
            shutil.copy2(src, dst)
            copied += 1
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "image", "label", "cohort", "device", "patient", "src"])
        for split, name, label, cohort in records:
            pt, dev = meta(name, cohort)
            w.writerow([split, name, label, cohort, dev, pt, os.path.join(img_dir[cohort], name)])
    print(f"\ncopied {copied} files -> {args.out}")


if __name__ == "__main__":
    main()
