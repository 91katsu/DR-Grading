"""Crop each fundus photo to the retinal disc, pad to square and resize.

Input : an ImageFolder tree (e.g. Fundus_Dataset/DR5/{train,val,test}/{0..4}/*.png)
Output: the same tree under --dst with every image as a <size>x<size> PNG.

The disc bounding box is found from row/column profiles of a brightness mask, so small
overlays (date stamp, "R Color" text in Canon images) do not enlarge the box.
"""
import argparse
import csv
import os
from multiprocessing import Pool

import cv2
import numpy as np


def fundus_bbox(img, thresh=15, min_frac=0.08):
    gray = img.max(axis=2)  # max over channels; the dark background is ~0 in all channels
    mask = gray > thresh
    h, w = mask.shape
    cols = np.where(mask.sum(axis=0) > min_frac * h)[0]
    rows = np.where(mask.sum(axis=1) > min_frac * w)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, w, h
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1


def process(job):
    src, dst, size = job
    img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)  # handles non-ASCII paths
    if img is None:
        return src, None
    x0, y0, x1, y1 = fundus_bbox(img)
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=img.dtype)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    out = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ok, buf = cv2.imencode(".png", out)
    buf.tofile(dst)
    return src, (img.shape[1], img.shape[0], x0, y0, x1, y1)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(root, "Fundus_Dataset", "DR5"))
    ap.add_argument("--dst", default=os.path.join(root, "Fundus_Dataset", "DR5_512"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = []
    for split in ("train", "val", "test"):
        for cls in sorted(os.listdir(os.path.join(args.src, split))):
            d = os.path.join(args.src, split, cls)
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    jobs.append((os.path.join(d, f), os.path.join(args.dst, split, cls, os.path.splitext(f)[0] + ".png"), args.size))

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
    src_manifest = os.path.join(args.src, "manifest.csv")
    if os.path.exists(src_manifest):
        with open(src_manifest, encoding="utf-8") as a, open(os.path.join(args.dst, "manifest.csv"), "w", encoding="utf-8") as b:
            b.write(a.read())
    print(f"done: {len(rows)} ok, {len(failed)} failed", failed[:10])


if __name__ == "__main__":
    main()
