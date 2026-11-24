"""Fundus_FDR26 (3000 jpg + metadata_classification_DR.xlsx) -> the same layout / preprocessing as APTOS2019_512,
for stage-1 training before the DR5 fine-tune.

Input  (--src): images/<image_id>.jpg and metadata_classification_DR.xlsx (sheet "Toutes":
                image_id, classe 0-4, grade_DR, sexe, age, type_diabete). image_id = 8-char patient code + 2-digit index.
Output (--dst): {train,val,test}/{0..4}/<image_id>.png  (crop retinal disc, pad square, resize, as preprocess_fundus.py)
                manifest.csv  (split, image, label, cohort=fdr26, device=FDR26, patient, group, src, sex, age, diabetes_type)
                crop_boxes.csv, removed.csv (every image / row left out and why),
                near_duplicates.csv (every ORB pair with >= 15 inliers, for checking)

Duplicates (byte-identical files, MD5), rule A as tools/dedup_dataset.py:
  same grade for every copy -> keep the first copy (sorted by name); grades disagree -> drop every copy.
Near duplicates (same eye photographed again / re-encoded, often under ANOTHER patient code): all-pairs ORB matching
       on the preprocessed images (500 keypoints at 256 px, RANSAC similarity transform); a pair with
       >= --near_dup_inliers inliers (default 25; visually: < 25 different eyes, >= 30 the same eye) is the same eye.
       Near duplicates are all kept (different visits may have different grades); they only constrain the split.
Split: by group = patient code, merged over exact and near duplicates (union-find), so no patient / eye is in two
       splits; stratified by the group's worst grade; --val_frac / --test_frac of the images, fixed --seed.
--test_frac 0: train_dr.py still needs <dst>/test, so test/ = hard links to the val images (NOT an independent test
       set; manifest.csv lists them as split=val only).

Downloaded data: run with -I from outside the data folder, e.g.
  ~/anaconda3/bin/python -I tools/prepare_fdr26.py --src /mnt/10TB_2024/Katsu/data/DR/Fundus_FDR26 \
      --dst /mnt/10TB_2024/Katsu/data/DR/Fundus_FDR26_512
(needs pandas + openpyxl + opencv: the base env has them, the dr env lacks openpyxl)
"""
import argparse
import csv
import hashlib
import os
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I does not add the script folder
from preprocess_fundus import process  # noqa: E402

SPLITS = ("train", "val", "test")


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _orb(path):
    import cv2
    import numpy as np
    g = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (256, 256), interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    mask = np.zeros_like(g)
    cv2.circle(mask, (128, 128), 118, 255, -1)
    k, d = cv2.ORB_create(500).detectAndCompute(g, mask)
    return np.float32([x.pt for x in k]), d


_F = None


def _orb_init(paths):
    import cv2
    global _F
    cv2.setNumThreads(1)
    _F = [_orb(p) for p in paths]


def _orb_row(i, min_report=15):
    import cv2
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    pa, da = _F[i]
    out = []
    if da is None:
        return out
    for j in range(i + 1, len(_F)):
        pb, db = _F[j]
        if db is None:
            continue
        mt = bf.match(da, db)
        if len(mt) < 20:
            continue
        _, inl = cv2.estimateAffinePartial2D(pa[[x.queryIdx for x in mt]], pb[[x.trainIdx for x in mt]],
                                             method=cv2.RANSAC, ransacReprojThreshold=3)
        n = int(inl.sum()) if inl is not None else 0
        if n >= min_report:
            out.append((i, j, n))
    return out


def near_duplicates(paths, workers):
    """all pairs (i, j, inliers) with >= 15 ORB inliers"""
    with Pool(workers, initializer=_orb_init, initargs=(paths,)) as pool:
        return [r for rs in pool.imap_unordered(_orb_row, range(len(paths)), chunksize=4) for r in rs]


def stratified_split(groups, worst, fracs, seed):
    """groups: {gid: [image, ...]}, worst: {gid: worst grade} -> {gid: split}.
    Per worst-grade stratum (shuffled with seed), each group goes to the split furthest below its target share."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for g in sorted(groups):
        by[worst[g]].append(g)
    out = {}
    for k in sorted(by):
        gs = by[k]
        rng.shuffle(gs)
        n = sum(len(groups[g]) for g in gs)
        acc = Counter()
        for g in gs:
            s = min((sp for sp in SPLITS if fracs[sp] > 0), key=lambda sp: acc[sp] / (fracs[sp] * n))
            out[g] = s
            acc[s] += len(groups[g])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--labels", default=None, help="default <src>/metadata_classification_DR.xlsx (.csv also accepted)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--near_dup_inliers", type=int, default=25, help="ORB inliers for 'same eye' (0 = no scan)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if os.path.exists(args.dst):
        sys.exit(f"{args.dst} exists: delete it or use another --dst")

    import pandas as pd
    lab_path = args.labels or os.path.join(args.src, "metadata_classification_DR.xlsx")
    df = pd.read_csv(lab_path) if lab_path.endswith(".csv") else pd.read_excel(lab_path)
    df["image_id"] = df["image_id"].astype(str).str.strip()
    assert df["image_id"].is_unique, "image_id not unique in the label file"
    assert df["classe"].between(0, 4).all(), "classe outside 0-4"
    meta = df.set_index("image_id")

    img_dir = os.path.join(args.src, "images")
    files = {os.path.splitext(f)[0]: os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    removed = []  # image, label, reason, detail
    for s in sorted(set(files) - set(meta.index)):
        removed.append([s, "", "no_label", "file has no row in the label file"])
    for s in sorted(set(meta.index) - set(files)):
        removed.append([s, int(meta.classe[s]), "no_image", "row has no image file"])
    stems = sorted(set(files) & set(meta.index))

    # ---- exact duplicates (rule A)
    with Pool(args.workers) as pool:
        hashes = dict(zip(stems, pool.map(md5, [files[s] for s in stems], chunksize=16)))
    dup = defaultdict(list)
    for s in stems:
        dup[hashes[s]].append(s)
    keep = []
    for h, v in dup.items():
        grades = {int(meta.classe[s]) for s in v}
        if len(v) == 1:
            keep.append(v[0])
        elif len(grades) == 1:
            keep.append(v[0])
            removed += [[s, int(meta.classe[s]), "duplicate_same_grade", f"identical to {v[0]} (kept)"] for s in v[1:]]
        else:
            removed += [[s, int(meta.classe[s]), "duplicate_conflicting_grade",
                         "identical to " + ", ".join(f"{x} ({int(meta.classe[x])})" for x in v if x != s)] for s in v]
    keep.sort()

    # ---- preprocess every kept image into a staging folder (needed by the near-duplicate scan)
    stage = os.path.join(args.dst, "_staging")
    os.makedirs(stage)
    jobs = [(files[s], os.path.join(stage, s + ".png"), args.size) for s in keep]
    failed, rows = [], []
    with Pool(args.workers) as pool:
        for i, (src, info) in enumerate(pool.imap_unordered(process, jobs, chunksize=8), 1):
            if info is None:
                failed.append(src)
            else:
                rows.append([os.path.relpath(src, args.src), *info])
            if i % 500 == 0:
                print(f"preprocess {i}/{len(jobs)}", flush=True)
    if failed:
        sys.exit(f"preprocessing failed: {failed[:10]}")

    # ---- near duplicates (ORB)
    pairs = []
    if args.near_dup_inliers > 0:
        print("near-duplicate scan (all pairs, ORB) ...", flush=True)
        pairs = near_duplicates([os.path.join(stage, s + ".png") for s in keep], args.workers)
    same_eye = [(keep[i], keep[j], n) for i, j, n in pairs if n >= args.near_dup_inliers]

    # ---- groups: patient code, merged over exact and near duplicates
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    patient = lambda s: s[:-2]  # noqa: E731
    for s in keep:
        find(patient(s))
    for v in dup.values():
        for s in v[1:]:
            parent[find(patient(s))] = find(patient(v[0]))
    for a, b, _ in same_eye:
        parent[find(patient(a))] = find(patient(b))
    groups = defaultdict(list)
    for s in keep:
        groups[find(patient(s))].append(s)
    gid = {g: f"g{k:04d}" for k, g in enumerate(sorted(groups, key=lambda g: min(groups[g])))}
    worst = {g: max(int(meta.classe[s]) for s in v) for g, v in groups.items()}
    fracs = {"train": 1 - args.val_frac - args.test_frac, "val": args.val_frac, "test": args.test_frac}
    split_of = stratified_split(groups, worst, fracs, args.seed)

    # ---- move into {split}/{label}/
    manifest = []
    for g, v in groups.items():
        for s in v:
            sp, y, r = split_of[g], int(meta.classe[s]), meta.loc[s]
            dst = os.path.join(args.dst, sp, str(y), s + ".png")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(os.path.join(stage, s + ".png"), dst)
            manifest.append([sp, s + ".png", y, "fdr26", "FDR26", patient(s), gid[g], files[s],
                             r.get("sexe", ""), r.get("age", ""), r.get("type_diabete", "")])
    os.rmdir(stage)
    manifest.sort(key=lambda m: (SPLITS.index(m[0]), m[1]))
    if args.test_frac == 0:  # train_dr.py needs a test folder: hard links to val
        for m in manifest:
            if m[0] == "val":
                a = os.path.join(args.dst, "val", str(m[2]), m[1])
                b = os.path.join(args.dst, "test", str(m[2]), m[1])
                os.makedirs(os.path.dirname(b), exist_ok=True)
                os.link(a, b)

    with open(os.path.join(args.dst, "crop_boxes.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "orig_w", "orig_h", "x0", "y0", "x1", "y1"])
        w.writerows(sorted(rows))
    with open(os.path.join(args.dst, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "image", "label", "cohort", "device", "patient", "group", "src", "sex", "age",
                    "diabetes_type"])
        w.writerows(manifest)
    with open(os.path.join(args.dst, "removed.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "label", "reason", "detail"])
        w.writerows(sorted(removed, key=lambda r: (r[2], r[0])))

    sp_of = {m[1][:-4]: m[0] for m in manifest}
    with open(os.path.join(args.dst, "near_duplicates.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "inliers", "same_eye", "grade_a", "grade_b", "patient_a", "patient_b", "split_a", "split_b"])
        for i, j, n in sorted(pairs, key=lambda p: -p[2]):
            a, b = keep[i], keep[j]
            w.writerow([a, b, n, int(n >= args.near_dup_inliers), int(meta.classe[a]), int(meta.classe[b]),
                        patient(a), patient(b), sp_of[a], sp_of[b]])
    print(f"near duplicates (>= {args.near_dup_inliers} inliers): {len(same_eye)} pairs, "
          f"{sum(patient(a) != patient(b) for a, b, _ in same_eye)} across patient codes, "
          f"{sum(int(meta.classe[a]) != int(meta.classe[b]) for a, b, _ in same_eye)} with different grades")
    print(f"label rows {len(meta)}, image files {len(files)}, matched {len(stems)}, kept {len(keep)}, "
          f"groups {len(groups)}")
    print("removed:", dict(Counter(r[2] for r in removed)))
    for sp in SPLITS:
        c = Counter(m[2] for m in manifest if m[0] == sp)
        if c:
            print(f"{sp:5} {sum(c.values()):5}  grade 0-4 {[c[g] for g in range(5)]}  "
                  f"groups {sum(1 for g in groups if split_of[g] == sp)}")
    if args.test_frac == 0:
        print("test = hard links to val (no independent test set)")
    print(f"done: {len(rows)} images")


if __name__ == "__main__":
    main()
