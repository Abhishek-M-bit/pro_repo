import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import json

SENSOR_FEATURES = [
    "LOAD_PCT", "ECT", "MAP", "RPM", "VSS", "IAT", 
    "MAF", "FRP", "BARO", "VPWR", "AAT"
]

DTC_FEATURES = [
    "P0000", "P0562", "P0113", "P0102", "P0403", "P0404", 
    "P2562", "P0234", "P2015", "P2009", "P0107", "P0069", 
    "P0089", "P0406"
]

CATEGORICAL_FEATURES = ["Mode"]

class PreprocessingPipeline:
    def __init__(self, train_path, test_path, output_dir, models_dir):
        self.train_path = train_path
        self.test_path = test_path
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.scaler = StandardScaler()
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def load_data(self):
        print("Loading datasets...")
        train_df = pd.read_excel(self.train_path)
        test_df = pd.read_excel(self.test_path)
        return train_df, test_df
        
    def validate_numeric(self, df, name="Dataset"):
        print(f"Validating {name}...")
        report = {}
        missing = df[SENSOR_FEATURES].isna().sum().sum()
        infs = np.isinf(df[SENSOR_FEATURES]).sum().sum()
        
        report['missing_values'] = int(missing)
        report['infinite_values'] = int(infs)
        
        # Check for strings disguised as numbers
        non_numeric_cols = []
        for col in SENSOR_FEATURES:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        report['non_numeric_cols'] = non_numeric_cols
        return report

    def extract_dtcs(self, row):
        active = []
        for dtc in DTC_FEATURES:
            if row.get(dtc, 0) == 1:
                active.append(dtc)
                
        # Handle P0000 semantics: 
        # If P0000 is 1 but there are other faults, P0000 might just be a default flag or logging artifact.
        # If active only has P0000, then it's 'No Fault'.
        # Let's standardize: if P0000 is present with others, we still record it but the signature shows the combo.
        if "P0000" in active and len(active) > 1:
            pass # Keep it to reflect raw data
        elif len(active) == 0:
            # If no DTC is 1, maybe it's implicitly P0000? 
            # We will just leave it empty.
            pass
            
        active.sort()
        signature = "|".join(active) if active else "NONE"
        return active, signature

    def process_dtcs(self, df):
        # Create dtc_signature and active_dtcs
        res = df.apply(self.extract_dtcs, axis=1)
        df['active_dtcs'] = [x[0] for x in res]
        df['dtc_signature'] = [x[1] for x in res]
        return df

    def deduplicate(self, train_df, test_df):
        orig_train_len = len(train_df)
        train_df_clean = train_df.drop_duplicates(keep='first').copy()
        dupes_removed = orig_train_len - len(train_df_clean)
        
        # Test duplicates calculation (without removing)
        test_dupes = test_df.duplicated().sum()
        
        return train_df_clean, dupes_removed, test_dupes

    def evaluation_cleaning(self, train_df, test_df):
        # Identify overlaps
        train_str = train_df.astype(str).apply(lambda x: ''.join(x), axis=1)
        test_str = test_df.astype(str).apply(lambda x: ''.join(x), axis=1)
        
        train_set = set(train_str)
        test_set = set(test_str)
        
        overlaps = train_set.intersection(test_set)
        return len(overlaps)

    def run(self):
        train_df, test_df = self.load_data()
        
        orig_train_len = len(train_df)
        orig_test_len = len(test_df)
        
        train_val = self.validate_numeric(train_df, "Train")
        test_val = self.validate_numeric(test_df, "Test")
        
        train_clean, train_dupes_removed, test_dupes = self.deduplicate(train_df, test_df)
        overlaps = self.evaluation_cleaning(train_clean, test_df)
        
        # DTC handling
        train_clean = self.process_dtcs(train_clean)
        test_clean = self.process_dtcs(test_df.copy()) # We don't remove dupes from test, just process features
        
        # Scale
        print("Fitting scaler on training data...")
        train_clean[SENSOR_FEATURES] = self.scaler.fit_transform(train_clean[SENSOR_FEATURES])
        test_clean[SENSOR_FEATURES] = self.scaler.transform(test_clean[SENSOR_FEATURES])
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, 'preprocessor.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save processed data
        train_out = os.path.join(self.output_dir, 'train_clean.csv')
        test_out = os.path.join(self.output_dir, 'test_clean.csv')
        
        train_clean.to_csv(train_out, index=False)
        test_clean.to_csv(test_out, index=False)
        
        # Analyze P0000 semantics
        p0000_only = train_clean[train_clean['dtc_signature'] == 'P0000']
        p0000_mixed = train_clean[train_clean['active_dtcs'].apply(lambda x: 'P0000' in x and len(x) > 1)]
        no_dtcs = train_clean[train_clean['dtc_signature'] == 'NONE']
        
        # Report
        report_content = f"""# Preprocessing Data Quality Report

## Row Counts & Deduplication
- **Original Train Rows**: {orig_train_len}
- **Cleaned Train Rows**: {len(train_clean)}
- **Train Duplicates Removed**: {train_dupes_removed}
- **Original Test Rows**: {orig_test_len} (Unmodified, used for evaluation)
- **Test Duplicates (Not removed)**: {test_dupes}
- **Train/Test Overlaps Detected**: {overlaps}

## Features
- **Sensor Features ({len(SENSOR_FEATURES)})**: {', '.join(SENSOR_FEATURES)}
- **DTC Features ({len(DTC_FEATURES)})**: {', '.join(DTC_FEATURES)}
- **Categorical Features**: {', '.join(CATEGORICAL_FEATURES)}

## Validation
- **Train Missing/Inf**: Missing={train_val['missing_values']}, Inf={train_val['infinite_values']}
- **Test Missing/Inf**: Missing={test_val['missing_values']}, Inf={test_val['infinite_values']}

## P0000 Semantics & Active DTCs
We extracted active DTCs per row into `active_dtcs` (list) and `dtc_signature` (string).
- Rows with ONLY P0000: {len(p0000_only)}
- Rows with P0000 AND other faults: {len(p0000_mixed)}
- Rows with NO faults (all 0s): {len(no_dtcs)}

**Decision on P0000:**
P0000 is included in the `active_dtcs` list if it equals 1. If it appears alongside other DTCs, we preserve it in the signature (e.g., `P0000|P0102`) to ensure deterministic representation without losing raw data fidelity. If it is the only fault, the signature is `P0000`. If no DTCs are 1, the signature is `NONE`.

## Mode Handling
Mode is treated strictly as a categorical variable. It has not been subjected to the numerical scaler.
Observed Mode Values in Train: {train_clean['Mode'].unique().tolist()}

## Scaling
- **Scaler Used**: StandardScaler from scikit-learn
- **Fit strategy**: Fitted ONLY on the deduplicated training data. Applied to both train and test.
- **Artifact**: Saved to `models/preprocessor.joblib`

## Output Files
- Processed Train: `data/processed/train_clean.csv`
- Processed Test: `data/processed/test_clean.csv`
"""
        report_path = os.path.join(self.output_dir, 'preprocessing_report.md')
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        print("Preprocessing complete!")

if __name__ == "__main__":
    # Configurable paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    train_path = os.path.join(base_dir, "MASTER_TRAIN_X.xlsx")
    test_path = os.path.join(base_dir, "MASTER_TEST_X.xlsx")
    
    ml_engine_dir = os.path.join(base_dir, "ml_engine")
    output_dir = os.path.join(ml_engine_dir, "data", "processed")
    models_dir = os.path.join(ml_engine_dir, "models")
    
    pipeline = PreprocessingPipeline(train_path, test_path, output_dir, models_dir)
    pipeline.run()
