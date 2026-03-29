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

## Trial 06 — Cross-City Generalisation Fixes
**Date:** 2026-03-15
**Root cause addressed:** Official evaluation returned macro F1=0.32 vs internal 0.57.
Diagnosis: severe cross-city distribution shift (Freetown tropical climate vs South American
training cities), compounded by two bugs in the validation pipeline.

**Changes:**
1. **CRITICAL BUG FIX — imputer missing from validation pipeline**
   - Previously: `sc.transform(val)` was called without first running `imputer.transform(val)`
   - NaN values from raster edge pixels in Freetown were passed directly to XGBoost
   - Fix: `val_arr = imputer.transform(val)` then `val_transformed = sc.transform(val_arr)`
2. **Removed SMOTE**
   - SMOTE generates synthetic samples that mimic the Santiago+Rio spectral distribution,
     making the model _more_ city-specific and _less_ generalisable to Freetown
   - Replaced with inverse-frequency `sample_weight` in XGBoost.fit()
3. **Reduced model complexity (anti-overfitting)**
   - `max_depth`: 7 → 4
   - `min_child_weight`: 3 → 5
   - `reg_lambda`: 1 → 2.0 (L2 regularisation)
   - `reg_alpha`: 0 → 0.5 (L1 sparsity)
   - `gamma`: 0.1 → 0.2
   - `colsample_bytree`: 0.8 → 0.7
4. **Added Leave-One-City-Out CV diagnostic**
   - Train on Chile → test on Brazil and vice versa
   - More realistic proxy for cross-city generalisation than same-city train/test split
5. **Replaced RandomizedSearchCV** (timed out with buffered data) with hardcoded
   generalisation-focused params

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| High | 0.40 | 0.28 | 0.33 | 5087 |
| Low | 0.42 | 0.36 | 0.39 | 3018 |
| Medium | 0.42 | 0.56 | 0.48 | 6000 |
| **Macro avg** | 0.41 | 0.40 | **0.40** | 14105 |
| **Accuracy** | | | **0.42** | 14105 |

> Result: **+0.08 macro F1 over official Trial 05 score** (+25% relative improvement).
> Imputer bug fix and SMOTE removal were the primary drivers.
> High class remains hardest — recall of only 0.28 means the model misses most hotspots.

---
## Trial 08 — New GeoTIFFs: Albedo + BSI + FVC + GLCM Texture + Per-City Normalisation
**Date:** 2026-03-29
**Changes:**
1. **Regenerated GeoTIFFs** — new 7-band layout replacing 6-band:
   - Dropped: UI (redundant with NDBI in S2), SAVI (semi-arid calibration, bad for tropical Freetown)
   - Added: **Albedo** (Bonafoni & Sekertekin 2020 formula), **BSI** (Bare Soil Index), **FVC** (Fractional Vegetation Cover)
2. **Albedo** — physically grounded heat absorption proxy; ranked #1 most transferable feature in cross-city UHI literature
3. **BSI** — explicitly designed for mixed bare soil / informal settlement roofing (directly relevant to Freetown)
4. **FVC** — normalises phenological offset between temperate Santiago/Rio and tropical Freetown (year-round green canopy)
5. **GLCM texture** — 4 features (contrast, homogeneity, energy, correlation) from NDBI band, 150m patch
6. **Per-city spectral normalisation** — each city's spectral features z-scored by own mean/std before training and for Freetown validation
7. **Building density caching** — saves computed density to CSV, eliminating 20-min recomputation on each run
8. **Total features: 14** (8 spectral + 2 density + 4 GLCM)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| High | 0.64 | 0.65 | 0.64 | 4893 |
| Low | 0.66 | 0.58 | 0.62 | 4281 |
| Medium | 0.55 | 0.61 | 0.58 | 4547 |
| **Macro avg** | 0.62 | 0.61 | **0.61** | 13721 |
| **Accuracy** | | | **0.61** | 13721 |

> Internal result: **+0.04 macro F1 over Trial 05 internal** (0.57 → 0.61).
> High class recall jumped from 0.28 → 0.65 — Albedo and BSI are the likely drivers.
> Official Freetown score pending upload.

---
## Trial 09 — Spatial Neighbourhood Features + Local Deviation Features
**Date:** 2026-03-29
**Changes:**
1. **500m neighbourhood means** (7 new features: neigh_NDVI, neigh_NDBI, neigh_NDWI, neigh_MNDWI, neigh_Albedo, neigh_BSI, neigh_FVC)
   - Average spectral value within a 500m window around each point
   - Captures local urban context — is this point in a generally hot zone or an isolated hotspot?
2. **Local deviation features** (4 new features: dev_NDVI, dev_NDBI, dev_Albedo, dev_BSI)
   - Local (50m) minus neighbourhood (500m) value
   - Self-normalising: already city-relative, no per-city normalisation needed
   - A High UHI point should have higher NDBI and lower Albedo than its surroundings regardless of city
3. **Efficient extraction**: both scales computed from a single 500m raster read per point (inner 50m patch sliced from large window)
4. **Total features: 14 → 25**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| High | 0.75 | 0.73 | 0.74 | 4893 |
| Low | 0.78 | 0.71 | 0.74 | 4292 |
| Medium | 0.61 | 0.68 | 0.64 | 4549 |
| **Macro avg** | 0.71 | 0.71 | **0.71** | 13734 |
| **Accuracy** | | | **0.71** | 13734 |

> Internal result: **+0.10 macro F1 over Trial 08** (0.61 → 0.71). Biggest single-trial gain so far.
> Spatial context was the missing piece — the deviation features in particular capture
> whether a point is hotter/cooler than its own surroundings, which transfers across cities.
> Official Freetown score pending upload.

---
