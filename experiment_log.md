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

## Trial 03 — Median Mosaic GeoTIFFs (cloud-free composite)
**Date:** 2026-03-13
**Changes:**
- Replaced single-date GeoTIFFs with **90-day median mosaics** generated via Planetary Computer API
- Cloud masking applied using Sentinel-2 SCL band (removed no-data, saturated, shadows, clouds)
- Santiago: 42 scenes → 3897×3897 px | Rio: 5 scenes → 4343×2673 px | Freetown: 36 scenes → 5567×4454 px
- Features unchanged: NDVI, NDBI, NDWI, MNDWI + building_density_100m

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| High | 0.52 | 0.55 | 0.54 |
| Low | 0.55 | 0.51 | 0.53 |
| Medium | 0.48 | 0.49 | 0.48 |
| **Macro avg** | 0.52 | 0.52 | **0.52** |
| **Accuracy** | | | **0.5162** |

> Result: No improvement. Cleaner imagery did not help — the bottleneck is likely the model itself (features, algorithm, or class boundary definition), not data quality.

---

## Trial 04 — XGBoost Classifier
**Date:** 2026-03-13
**Changes:**
- Replaced Random Forest with **XGBoost** (`n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`)
- Data unchanged: median mosaic GeoTIFFs, same 5 features

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| High | 0.54 | 0.56 | 0.55 |
| Low | 0.57 | 0.52 | 0.54 |
| Medium | 0.49 | 0.52 | 0.50 |
| **Macro avg** | 0.53 | 0.53 | **0.53** |
| **Accuracy** | | | **0.5321** |

> Result: Small but consistent improvement over RF (+0.01 macro F1). Medium class improved most (+0.02). XGBoost is now the best model.

---

## Trial 05 — Feature Engineering + SMOTE + Hyperparameter Tuning
**Date:** 2026-03-13
**Changes:**
- **6-band GeoTIFFs**: Added UI (B12-B08)/(B12+B08) and SAVI ((B08-B04)/(B08+B04+0.5))*1.5
- **IBI**: Derived from existing indices — `[2*NDBI-(NDVI+NDWI)] / [2*NDBI+(NDVI+NDWI)]`
- **50m building density**: Added alongside 100m (9 features total vs 5)
- **SMOTE**: Balanced training classes (11,286 each: High/Low/Medium)
- **XGBoost hyperparameter tuning**: RandomizedSearchCV (30 iterations, 3-fold CV)
  - Best params: `n_estimators=500, max_depth=7, lr=0.05, subsample=0.8, gamma=0.1, min_child_weight=3`
  - Best CV macro F1: 0.5688

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| High | 0.58 | 0.60 | 0.59 |
| Low | 0.62 | 0.57 | 0.59 |
| Medium | 0.51 | 0.54 | 0.52 |
| **Macro avg** | 0.57 | 0.57 | **0.57** |
| **Accuracy** | | | **0.5685** |

> Result: **+0.04 macro F1 over Trial 04** (+8% relative improvement). New features (UI, SAVI, IBI, 50m density) + SMOTE drove the majority of the gain. Medium class remains hardest to classify.

---

## Trial 06 — *(next experiment)*
