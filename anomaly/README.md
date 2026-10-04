# Anomaly Detection

## Purpose
The Anomaly Detection component aims to evaluate whether a current vehicle operating condition (represented by 11 numerical sensor parameters) is unusual when compared strictly against the historical operating conditions present in the training dataset.

## Available Data Limitations
The available dataset (`MASTER_TRAIN_X`) provides point-in-time vehicle telemetry and DTCs, but crucially **lacks**:
- Reliable timestamps for sequence/frequency analysis.
- Customer complaints or inspection findings.
- Verified repair outcomes or component failure labels.
- Confirmed "healthy" vs "broken" fleet annotations.

Because of this, we cannot train a supervised classification model to predict "failure." We can only identify numerical deviations from the steady-state norm.

## Algorithms Considered
1. **Statistical Thresholding (Baseline)**: Simple Z-score or percentile-based bounding box. Highly transparent, but struggles with multi-dimensional covariance (e.g., high RPM + low Speed might be normal, but high RPM + high Speed is normal too; simple thresholds miss these relationships).
2. **Isolation Forest**: An unsupervised ensemble tree method that isolates anomalies rather than profiling normal points. Excellent for high-dimensional, unlabeled tabular data.
3. **One-Class SVM**: Effective but computationally expensive and highly sensitive to hyperparameter tuning without validation data.

## Selected Algorithm
**Isolation Forest** was selected. It efficiently handles the 11-dimensional feature space and naturally isolates conditions that are numerically rare or out-of-distribution without assuming a Gaussian distribution of the data. 

## Training Procedure
- **Data**: Trained solely on the 27,820 deduplicated rows of `train_clean.csv`.
- **Features**: The 11 continuous sensor variables (`LOAD_PCT`, `RPM`, `ECT`, etc.). Categorical `Mode` and binary `DTCs` are excluded from the anomaly forest.
- **Preprocessing**: Sensor values were scaled using the pre-fitted `StandardScaler`.
- **Contamination**: Set conservatively at `0.01` (1%), meaning the model expects true extreme anomalies to be rare.
- **Artifacts**: The fitted model is saved as `anomaly_model.joblib`. We also save `feature_stats.joblib` (5th and 95th percentiles of the training distribution) for explainability.

## Anomaly Score Meaning
The anomaly score output by the model is the negated Isolation Forest decision function. 
- **Score < 0**: Normal operating condition. The condition falls well within the dense regions of historical data.
- **Score > 0**: Anomalous condition. The condition was easily isolated and lies outside normal bounds.

*Important: This score is NOT a probability (e.g., it is not bounded 0-1) and does not represent the probability of component failure.*

## Threshold / Severity Logic
- `score < 0`: **Normal** (Typical operating state)
- `0 <= score < 0.1`: **Moderate** (Unusual, but not extreme)
- `score >= 0.1`: **High** (Highly unusual / Out of bounds)

## Feature Contributions
Because Isolation Forest is an ensemble method, extracting exact feature contributions per instance is complex. Instead, we use a transparent statistical baseline for explainability: **Percentile Deviation**. 
If an anomaly is detected, we check the query's scaled features against the 5th and 95th percentiles of the training data. Any feature falling outside this band is flagged, and the magnitude of deviation is reported.

## Limitations & Disclaimer
**Anomaly detection does NOT prove root cause.**
This model simply identifies that the numerical sensor readings are historically rare. A high anomaly score could be due to a catastrophic sensor failure, an extreme but safe driving maneuver, or a harsh environmental condition not captured in the dataset. It provides a signal for the engineer to investigate, not a confirmed diagnosis.
