# UHI Model Experiment Log

Each trial records what changed and the out-of-sample classification results on the test split (30% holdout, stratified).

**Model:** Random Forest (`n_estimators=100`, `class_weight='balanced'`, `random_state=42`)
**Train cities:** Santiago (Chile) + Rio de Janeiro (Brazil)
**Validation city:** Freetown (Sierra Leone)

---

## Trial 01 — Baseline (3-band GeoTIFFs)
**Date:** 2026-03-13
**Changes:** Reference notebook as-is. Features: NDVI, NDBI, NDWI (3-band GeoTIFFs) + building_density_100m.

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| High | — | — | — |
| Low | — | — | — |
| Medium | — | — | — |
| **Macro avg** | — | — | **~0.52** |
| **Accuracy** | | | **0.5213** |

> Note: Original notebook reported ~0.5213 accuracy. Per-class F1 not recorded separately.

---

## Trial 02 — Add MNDWI (4-band GeoTIFFs)
**Date:** 2026-03-13
**Changes:**
- Switched to new 4-band GeoTIFFs (NDVI, NDBI, NDWI, MNDWI)
- Added `median_MNDWI` as a 5th predictor feature
- Vectorized `compute_building_density` using spatial join (performance fix)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| High | 0.53 | 0.57 | 0.55 |
| Low | 0.56 | 0.51 | 0.54 |
| Medium | 0.48 | 0.47 | 0.48 |
| **Macro avg** | 0.52 | 0.52 | **0.52** |
| **Accuracy** | | | **0.5202** |

> Result: No meaningful improvement over baseline. MNDWI alone did not add predictive value.

---

## Trial 03 — *(next experiment)*
