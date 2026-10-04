import os
import pandas as pd
import numpy as np
import joblib

def test_preprocessing():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    processed_dir = os.path.join(base_dir, "ml_engine", "data", "processed")
    models_dir = os.path.join(base_dir, "ml_engine", "models")
    
    train_out = os.path.join(processed_dir, 'train_clean.csv')
    test_out = os.path.join(processed_dir, 'test_clean.csv')
    scaler_path = os.path.join(models_dir, 'preprocessor.joblib')
    
    # 1. Preprocessing runs successfully / Outputs exist
    assert os.path.exists(train_out), "Train output not found"
    assert os.path.exists(test_out), "Test output not found"
    assert os.path.exists(scaler_path), "Scaler artifact not found"
    
    train_df = pd.read_csv(train_out)
    test_df = pd.read_csv(test_out)
    
    # 2. Output contains expected columns
    expected_cols = ["active_dtcs", "dtc_signature"]
    for c in expected_cols:
        assert c in train_df.columns, f"Missing {c} in train"
        assert c in test_df.columns, f"Missing {c} in test"
        
    # 3. No unexpected NaN or Inf values in sensors
    SENSOR_FEATURES = [
        "LOAD_PCT", "ECT", "MAP", "RPM", "VSS", "IAT", 
        "MAF", "FRP", "BARO", "VPWR", "AAT"
    ]
    assert train_df[SENSOR_FEATURES].isna().sum().sum() == 0, "NaNs found in train sensors"
    assert test_df[SENSOR_FEATURES].isna().sum().sum() == 0, "NaNs found in test sensors"
    
    assert np.isinf(train_df[SENSOR_FEATURES]).sum().sum() == 0, "Infs found in train sensors"
    
    # 4. Mode is not treated as a continuous sensor (should still be original values like 0, 1, 2)
    mode_vals = train_df['Mode'].unique()
    # If scaled, it would have decimals. If categorical, it should be integers usually.
    assert all(val in [0, 1, 2] for val in mode_vals), "Mode values seem modified/scaled."
    
    # 5. DTC signature generation is deterministic
    # We can check a few rows
    signatures = train_df['dtc_signature'].tolist()
    # "P0000|P0102" is correctly sorted (P0000 before P0102)
    for sig in signatures:
        if sig != "NONE":
            parts = sig.split("|")
            assert parts == sorted(parts), f"Signature {sig} is not sorted deterministically"
            
    # 6. Original raw datasets remain unchanged
    train_raw_path = os.path.join(base_dir, "MASTER_TRAIN_X.xlsx")
    orig_train = pd.read_excel(train_raw_path)
    assert 'dtc_signature' not in orig_train.columns, "Original train file was modified!"
    
    print("All preprocessing tests passed!")

if __name__ == "__main__":
    test_preprocessing()
