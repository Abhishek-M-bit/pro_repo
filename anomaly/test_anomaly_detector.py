import os
import pandas as pd
import numpy as np
import joblib
from anomaly_detector import AnomalyDetector

def test_anomaly_detector():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_path = os.path.join(base_dir, "ml_engine", "data", "processed", "train_clean.csv")
    scaler_path = os.path.join(base_dir, "ml_engine", "models", "preprocessor.joblib")
    model_dir = os.path.join(base_dir, "ml_engine", "models")
    
    detector = AnomalyDetector(model_dir=model_dir, scaler_path=scaler_path)
    
    # 7. Model loading is tested here implicitly if detector.model is populated
    assert detector.model is not None, "Model failed to load"
    assert detector.scaler is not None, "Scaler failed to load"
    
    df = pd.read_csv(data_path)
    scaler = joblib.load(scaler_path)
    
    # Get a real normal historical row
    row_0 = df.iloc[0]
    sensor_cols = detector.sensor_cols
    scaled_vals = row_0[sensor_cols].values.reshape(1, -1)
    orig_vals = scaler.inverse_transform(scaled_vals)[0]
    
    query_normal = {col: float(orig_vals[i]) for i, col in enumerate(sensor_cols)}
    
    # 1. Normal historical row
    res_normal = detector.detect(query_normal)
    
    # 5. Score generation & 6. Severity classification
    assert "anomaly_score" in res_normal
    assert "severity" in res_normal
    # We expect normal row to not be anomalous
    assert res_normal["severity"] == "normal"
    assert res_normal["anomaly"] is False
    
    # 8. Deterministic inference
    res_normal_2 = detector.detect(query_normal)
    assert res_normal["anomaly_score"] == res_normal_2["anomaly_score"]
    
    # 2. Artificially extreme operating condition
    query_extreme = query_normal.copy()
    for col in sensor_cols:
        query_extreme[col] = 999999.0 # Utterly absurd values for all sensors
    
    res_extreme = detector.detect(query_extreme)
    print("Normal score:", res_normal)
    print("Extreme score:", res_extreme)
    assert res_extreme["anomaly"] is True, "Failed to detect extreme anomaly"
    assert res_extreme["severity"] in ["moderate", "high"]
    
    # Check feature contributions
    feats = [f["feature"] for f in res_extreme["feature_contributions"]]
    assert "RPM" in feats
    assert "ECT" in feats
    
    # 3. Missing feature
    try:
        query_missing = query_normal.copy()
        del query_missing["MAP"]
        detector.detect(query_missing)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
        
    # 4. Invalid numerical value
    try:
        query_invalid = query_normal.copy()
        query_invalid["MAP"] = "NOT_A_NUMBER"
        detector.detect(query_invalid)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("All Anomaly Detector tests passed!")
    
    print("\n--- DEMO TEST ---")
    print("Normal Query Result:")
    print(res_normal)
    print("\nExtreme Query Result:")
    print(res_extreme)

if __name__ == "__main__":
    test_anomaly_detector()
