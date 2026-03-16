# Urban Heat Island Classification Project
## Week 7 Status Update – Group 14

---

### 1. Activities Worked This Week

This week the team focused on establishing a shared technical workflow and running initial model experiments.

First, the team organized a shared **GitHub repository** to centralize project code, datasets, and experiment outputs. This repository allows the team to track experiments and ensures that all members are working from the same baseline code. An **experiment log** was also created to document every trial with its changes, hyperparameters, and results.

Second, satellite data preprocessing was performed using **Sentinel-2 imagery**. A median mosaic approach was applied over a 90-day time window to reduce cloud noise and generate cleaner GeoTIFF inputs for feature extraction. Cloud masking was implemented using the Sentinel-2 SCL classification band to remove clouds, shadows, and no-data pixels before computing the median composite.

Finally, the team began model experimentation using the **XGBoost algorithm**, conducting five distinct trials to systematically evaluate the impact of preprocessing, feature engineering, class balancing, and hyperparameter tuning on prediction performance.

---

### 2. What Was Learned

During this week's experiments, the team observed that improvements in satellite preprocessing alone did not significantly increase the model's F1 score.

Although the median mosaic preprocessing produced cleaner imagery by reducing cloud noise, the model performance remained largely unchanged (macro F1 ~0.52 across Trials 01–03). This confirmed that preprocessing improvements alone were not sufficient to improve prediction accuracy.

Based on this observation, the team shifted focus toward **feature engineering and model optimization**. The key insight was that adding more informative spectral indices and addressing class imbalance had a much larger impact on performance than data quality improvements alone.

The team also reinforced the importance of maintaining a shared repository and experiment log so that different model trials can be tracked and compared efficiently, preventing duplicated effort.

---

### 3. How the Model Was Improved

After the initial preprocessing experiments, the team focused on improving the machine learning model through a combination of feature engineering, class balancing, and hyperparameter tuning.

**Algorithm Change (Trial 04)**
The Random Forest classifier was replaced with the **XGBoost classifier**, which produced a modest but consistent improvement, raising the macro F1 from ~0.52 to 0.53. XGBoost's iterative error-correction approach better handles overlapping class boundaries compared to averaging independent trees.

**Feature Engineering (Trial 05)**
The most significant improvements came from expanding the feature set from 5 to 9 predictors:

- **Urban Index (UI):** `(B12 - B08) / (B12 + B08)` — highlights human-made structures using SWIR-2 and NIR bands
- **Soil Adjusted Vegetation Index (SAVI):** `((B08 - B04) / (B08 + B04 + 0.5)) × 1.5` — corrects for soil brightness in areas with sparse vegetation
- **Index-based Built-up Index (IBI):** `[2×NDBI − (NDVI + NDWI)] / [2×NDBI + (NDVI + NDWI)]` — derived from existing indices to better isolate built-up areas from bare soil
- **50m building density buffer:** Added alongside the existing 100m buffer to capture localized structural density

New 6-band GeoTIFFs were generated via the Planetary Computer API to support the UI and SAVI calculations, incorporating bands B03, B04, B08, B11, and B12.

**Class Balancing with SMOTE**
The training set was balanced using **SMOTE (Synthetic Minority Over-sampling Technique)**, equalizing all three classes to 11,286 samples each. This directly addressed the class imbalance that was disproportionately hurting F1 scores on the Low and Medium classes.

**Hyperparameter Tuning**
A RandomizedSearchCV with 30 iterations and 3-fold cross-validation was used to identify optimal XGBoost parameters:
- `n_estimators=500, max_depth=7, learning_rate=0.05`
- `subsample=0.8, colsample_bytree=0.8, gamma=0.1, min_child_weight=3`

Through these combined improvements, the model achieved a **macro F1 score of 0.57** on the out-of-sample test set, representing a **+10% relative improvement** over the baseline. The prediction results from the improved model have been uploaded to the repository.

| Trial | Key Change | Macro F1 |
|-------|-----------|----------|
| 01 | Baseline RF, 3-band GeoTIFFs | 0.52 |
| 02 | +MNDWI (4-band GeoTIFFs) | 0.52 |
| 03 | Median mosaic preprocessing | 0.52 |
| 04 | XGBoost classifier | 0.53 |
| 05 | +UI, SAVI, IBI, 50m density, SMOTE, tuning | **0.57** |

---

### 4. Team Member Contributions

Although the project proposal defined different roles, the team decided to collaborate across tasks while maintaining primary responsibilities.

**Wen Xue**
- Organized the GitHub project board and repository structure
- Maintained documentation and experiment logs
- Coordinated team communication and status updates

**Smitha Balasubramanian**
- Conducted satellite data preprocessing using median mosaics via the Planetary Computer API
- Implemented spectral feature engineering (UI, SAVI, IBI indices)
- Implemented and trained the XGBoost model with SMOTE balancing and hyperparameter tuning

**Stephen Kao**
- Participated in discussions regarding model development and evaluation
- Assisted with reviewing preprocessing and modeling workflow
- Began exploring the value case related to urban heat impacts

**Sushma Bukkaperam**
- Participated in preprocessing review and documentation
- Reviewed the GitHub repository and ongoing model experiments

---

### 5. Plans for the Coming Week

Based on the current progress and team discussions, the team will focus on the following tasks during the next week:

1. Continue experimenting with the current modeling pipeline to achieve a **macro F1 score higher than 0.62**, focusing on spatial buffering of spectral features and texture-based features.
2. Implement **50m spatial buffer extraction** for spectral indices — replacing single-pixel nearest-neighbour extraction with mean values computed over a buffer window to reduce noise.
3. Explore **GLCM texture features** (Mean, Variance, Correlation) to help differentiate urban surface types with similar spectral signatures but different thermal properties.
4. Upload improved prediction outputs when a model achieves higher performance.
5. Continue documenting all model experiments and parameter changes in the GitHub experiment log.

The team will continue monitoring model performance and submit updated prediction files once improved models are developed.
