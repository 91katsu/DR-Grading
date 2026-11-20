# postprocess/ — copy of the deployed DR post-processing (ALFA)

Copied unmodified on 2026-11-20 from the deployment repo, service `postprocess` v2.1.9
(`deployment/postprocess/postprocess/*` and `deployment/postprocess/threshold_config.json`).
Used by `eval_mtm_vt_postprocess.py`. **Do not edit these files**: they must stay identical to the
deployed service. If the deployed version changes, copy the files again and update the table below.

| file | role | sha256 |
|---|---|---|
| `postprocess.py` | `postprocess()`: IoM-NMS 0.5 -> box to original image -> `contradict_DR5` -> lesion threshold filter | `4e2d5920...38966` |
| `ALFA.py` | `det2cls` (lesions -> DR grade) and `contradict_DR5` (classifier vs detection reconciliation) | `e21826cc...841ef` |
| `nms.py` | `iom_nms` | `6aadadbd...c2c` |
| `adjust_box.py` | `adjust_bbox` (remove padding, add crop offset) | `3d249569...bce` |
| `cls_prob_jye01.csv` / `Score_jye01.csv` | ALFA tables for type 0 (JYE01) | `ff146784...84e9` / `8cf78730...13e9` |
| `cls_prob.csv` / `Score.csv` | ALFA tables for type 1 (desktop) | `132f003b...4181` / `291673e5...ebc` |
| `threshold_config.json` | lesion score thresholds; type 0 by age group, type 1 single set | `da5b2c65...0a1f` |

Not deployed: `threshold_config_tuned.json` (sha256 `b57483d7...37a5`), copied from `deployment/postprocess/` on 2026-11-20,
only used with `eval_mtm_vt_postprocess.py --threshold_config postprocess/threshold_config_tuned.json`.
It differs from `threshold_config.json` only in type 0, age >= 50: IH (2) 0.2 -> 0.08, HE (8) 0.324 -> 0.06, LS (12) 0.359 -> 0.08.

The csv / json hashes of the deployed files equal the deployed `manifest.json`. `eval_mtm_vt_postprocess.py` prints the full sha256 of
every file into its `run_config.json`.

Notes
- `postprocess.py` opens the tables as `./postprocess/<table>.csv`, i.e. relative to the **repo root**;
  `eval_mtm_vt_postprocess.py` changes into the repo root before calling it.
- `ALFA.py` needs `pandas` (reads the tables).
- Not reproduced offline: the quality filter and the spot-light elimination (type 0) of the API,
  because their outputs are not part of the saved detections.
