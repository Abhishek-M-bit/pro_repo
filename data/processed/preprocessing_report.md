# Preprocessing Data Quality Report

## Row Counts & Deduplication
- **Original Train Rows**: 118477
- **Cleaned Train Rows**: 27820
- **Train Duplicates Removed**: 90657
- **Original Test Rows**: 33838 (Unmodified, used for evaluation)
- **Test Duplicates (Not removed)**: 26280
- **Train/Test Overlaps Detected**: 173

## Features
- **Sensor Features (11)**: LOAD_PCT, ECT, MAP, RPM, VSS, IAT, MAF, FRP, BARO, VPWR, AAT
- **DTC Features (14)**: P0000, P0562, P0113, P0102, P0403, P0404, P2562, P0234, P2015, P2009, P0107, P0069, P0089, P0406
- **Categorical Features**: Mode

## Validation
- **Train Missing/Inf**: Missing=0, Inf=0
- **Test Missing/Inf**: Missing=0, Inf=0

## P0000 Semantics & Active DTCs
We extracted active DTCs per row into `active_dtcs` (list) and `dtc_signature` (string).
- Rows with ONLY P0000: 6732
- Rows with P0000 AND other faults: 0
- Rows with NO faults (all 0s): 1338

**Decision on P0000:**
P0000 is included in the `active_dtcs` list if it equals 1. If it appears alongside other DTCs, we preserve it in the signature (e.g., `P0000|P0102`) to ensure deterministic representation without losing raw data fidelity. If it is the only fault, the signature is `P0000`. If no DTCs are 1, the signature is `NONE`.

## Mode Handling
Mode is treated strictly as a categorical variable. It has not been subjected to the numerical scaler.
Observed Mode Values in Train: [0, 2, 1]

## Scaling
- **Scaler Used**: StandardScaler from scikit-learn
- **Fit strategy**: Fitted ONLY on the deduplicated training data. Applied to both train and test.
- **Artifact**: Saved to `models/preprocessor.joblib`

## Output Files
- Processed Train: `data/processed/train_clean.csv`
- Processed Test: `data/processed/test_clean.csv`
