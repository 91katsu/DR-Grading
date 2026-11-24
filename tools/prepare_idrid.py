"""IDRiD grading images (Kaggle copy: Imagenes/IDRiD_xxx[test].jpg + idrid_labels.csv) -> the same layout /
preprocessing as Fundus_FDR26_512, as EXTRA TRAINING data (default: every image in train, no val / test), to be
merged with Fundus_FDR26_512 by tools/merge_datasets.py.

Input  (--src): Imagenes/<id_code>.jpg and idrid_labels.csv (id_code, diagnosis 0-4, Risk of macular edema 0-2;
                the empty extra columns are ignored). The Kaggle copy has 455 of the 516 official images.
Output (--dst): train/{0..4}/<id_code>.png
                manifest.csv  (same columns as Fundus_FDR26_512: split, image, label, cohort=idrid, device=IDRiD,
                               patient (= id_code, IDRiD has no patient ids), group, src, sex, age, diabetes_type
                               (empty)); the DME grade is in dme.csv
                crop_boxes.csv, removed.csv, near_duplicates.csv
Duplicates: exact (MD5), rule A (same grade -> keep the first copy, grades disagree -> drop every copy);
            near duplicates (ORB, >= --near_dup_inliers) are kept and only share a group id.

Downloaded data: run with -I from outside the data folder, e.g.
  ~/anaconda3/bin/python -I tools/prepare_idrid.py --src /mnt/10TB_2024/Katsu/data/DR/Imagenes \
      --dst /mnt/10TB_2024/Katsu/data/DR/IDRiD_512
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I does not add the script folder
from preprocess_fundus import process  # noqa: E402
from prepare_fdr26 import md5, near_duplicates  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--near_dup_inliers", type=int, default=40,
                    help="ORB inliers for 'same eye' (IDRiD: < 32 were different eyes, >= 88 the same)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if os.path.exists(args.dst):
        sys.exit(f"{args.dst} exists: delete it or use another --dst")

    labels = {}
    with open(os.path.join(args.src, "idrid_labels.csv"), encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        dme_col = next(c for c in rd.fieldnames if c.strip() == "Risk of macular edema")  # header has a trailing space
        for r in rd:
            s = r["id_code"].strip()
            if s:
                labels[s] = (int(r["diagnosis"]), int(r[dme_col]))
    assert all(0 <= g <= 4 for g, _ in labels.values())
    img_dir = os.path.join(args.src, "Imagenes")
    files = {os.path.splitext(f)[0]: os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    removed = [[s, "", "no_label", "file has no row in idrid_labels.csv"] for s in sorted(set(files) - set(labels))]
    removed += [[s, labels[s][0], "no_image", "row has no image file"] for s in sorted(set(labels) - set(files))]
    stems = sorted(set(files) & set(labels))

    with Pool(args.workers) as pool:
        hashes = dict(zip(stems, pool.map(md5, [files[s] for s in stems])))
    dup = defaultdict(list)
    for s in stems:
        dup[hashes[s]].append(s)
    keep = []
    for v in dup.values():
        grades = {labels[s][0] for s in v}
        if len(v) == 1 or len(grades) == 1:
            keep.append(v[0])
            removed += [[s, labels[s][0], "duplicate_same_grade", f"identical to {v[0]} (kept)"] for s in v[1:]]
        else:
            removed += [[s, labels[s][0], "duplicate_conflicting_grade",
                         "identical to " + ", ".join(f"{x} ({labels[x][0]})" for x in v if x != s)] for s in v]
    keep.sort()

    jobs = [(files[s], os.path.join(args.dst, "train", str(labels[s][0]), s + ".png"), args.size) for s in keep]
    with Pool(args.workers) as pool:
        res = pool.map(process, jobs, chunksize=4)
    failed = [s for s, info in res if info is None]
    if failed:
        sys.exit(f"preprocessing failed: {failed[:10]}")

    pairs = near_duplicates([j[1] for j in jobs], args.workers) if args.near_dup_inliers > 0 else []
    parent = {s: s for s in keep}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j, n in pairs:
        if n >= args.near_dup_inliers:
            parent[find(keep[i])] = find(keep[j])
    roots = sorted({find(s) for s in keep}, key=lambda r: min(s for s in keep if find(s) == r))
    gid = {r: f"idrid{k:04d}" for k, r in enumerate(roots)}

    with open(os.path.join(args.dst, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "image", "label", "cohort", "device", "patient", "group", "src", "sex", "age",
                    "diabetes_type"])
        w.writerows(["train", s + ".png", labels[s][0], "idrid", "IDRiD", s, gid[find(s)], files[s], "", "", ""]
                    for s in keep)
    with open(os.path.join(args.dst, "dme.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "dr", "dme"])
        w.writerows([s + ".png", *labels[s]] for s in keep)
    with open(os.path.join(args.dst, "crop_boxes.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "orig_w", "orig_h", "x0", "y0", "x1", "y1"])
        w.writerows(sorted([os.path.relpath(s, args.src), *info] for s, info in res))
    with open(os.path.join(args.dst, "removed.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "label", "reason", "detail"])
        w.writerows(sorted(removed, key=lambda r: (r[2], r[0])))
    with open(os.path.join(args.dst, "near_duplicates.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "inliers", "same_eye", "grade_a", "grade_b"])
        for i, j, n in sorted(pairs, key=lambda p: -p[2]):
            w.writerow([keep[i], keep[j], n, int(n >= args.near_dup_inliers), labels[keep[i]][0], labels[keep[j]][0]])

    same = [(keep[i], keep[j]) for i, j, n in pairs if n >= args.near_dup_inliers]
    c = Counter(labels[s][0] for s in keep)
    print(f"label rows {len(labels)}, image files {len(files)}, matched {len(stems)}, kept {len(keep)}")
    print("removed:", dict(Counter(r[2] for r in removed)))
    print(f"near duplicates (>= {args.near_dup_inliers} inliers): {len(same)} pairs, "
          f"{sum(labels[a][0] != labels[b][0] for a, b in same)} with different grades: {same}")
    print(f"train {len(keep)}  grade 0-4 {[c[g] for g in range(5)]}")


if __name__ == "__main__":
    main()
