import os
import sys
import pandas as pd
import joblib

# Add ml_engine to sys.path to allow imports from other submodules
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ml_engine_dir = os.path.join(base_dir, "ml_engine")
if ml_engine_dir not in sys.path:
    sys.path.append(ml_engine_dir)

from evidence.evidence_aggregator import EvidenceAggregator

def test_evidence_aggregator():
    data_path = os.path.join(ml_engine_dir, "data", "processed", "train_clean.csv")
    scaler_path = os.path.join(ml_engine_dir, "models", "preprocessor.joblib")
    anomaly_model_dir = os.path.join(ml_engine_dir, "models")
    rag_index_path = os.path.join(ml_engine_dir, "rag", "faiss.index")
    rag_meta_path = os.path.join(ml_engine_dir, "rag", "metadata.pkl")
    
    aggregator = EvidenceAggregator(data_path, scaler_path, anomaly_model_dir, rag_index_path, rag_meta_path)
    
    # Get a real normal historical row that has P0403 to test exact match and realistic ranges
    df = pd.read_csv(data_path)
    
    # Find a row with P0403 in its active_dtcs
    p0403_rows = df[df['dtc_signature'].str.contains('P0403', na=False)]
    if len(p0403_rows) > 0:
        row_real = p0403_rows.iloc[0]
    else:
        row_real = df.iloc[0] # Fallback
        
    scaler = joblib.load(scaler_path)
    sensor_cols = [
        "LOAD_PCT", "ECT", "MAP", "RPM", "VSS", "IAT", 
        "MAF", "FRP", "BARO", "VPWR", "AAT"
    ]
    
    scaled_vals = row_real[sensor_cols].values.reshape(1, -1)
    orig_vals = scaler.inverse_transform(scaled_vals)[0]
    
    import ast
    try:
        active_dtcs = ast.literal_eval(row_real["active_dtcs"]) if isinstance(row_real["active_dtcs"], str) else []
    except:
        active_dtcs = []
        
    # 1. Valid incident (Normal, Multiple DTCs)
    incident_normal = {col: float(orig_vals[i]) for i, col in enumerate(sensor_cols)}
    incident_normal["Mode"] = int(row_real["Mode"])
    incident_normal["active_dtcs"] = active_dtcs
    
    res_normal = aggregator.aggregate(incident_normal)
    
    # 13. Service history explicitly marked unavailable
    assert res_normal["service_history"]["available"] is False
    assert "No real service-history" in res_normal["service_history"]["reason"]
    
    # 7. Similar historical cases
    assert len(res_normal["similarity"]["top_cases"]) > 0
    
    # 6. Normal incident
    assert res_normal["anomaly"]["anomaly"] is False
    
    # 2. Multiple DTCs / 8. RAG exact match
    if "P0403" in active_dtcs:
        exact_matches = res_normal["rag"]["exact_dtc_matches"]
        assert any(m["DTC_code"] == "P0403" for m in exact_matches), "Missing exact RAG match for P0403"
        assert len(exact_matches) > 0
    
    # 3. Single DTC
    incident_single = incident_normal.copy()
    incident_single["active_dtcs"] = ["P0403"]
    res_single = aggregator.aggregate(incident_single)
    assert any(m["DTC_code"] == "P0403" for m in res_single["rag"]["exact_dtc_matches"])
    
    # 4. No DTC
    incident_none = incident_normal.copy()
    incident_none["active_dtcs"] = []
    res_none = aggregator.aggregate(incident_none)
    assert len(res_none["rag"]["exact_dtc_matches"]) == 0
    
    # 5. Anomalous incident
    incident_anomalous = incident_normal.copy()
    for col in sensor_cols:
        incident_anomalous[col] = 999999.0
    res_anomalous = aggregator.aggregate(incident_anomalous)
    assert res_anomalous["anomaly"]["anomaly"] is True
    
    # 10. Missing required sensor
    incident_missing = incident_normal.copy()
    del incident_missing["RPM"]
    res_missing = aggregator.aggregate(incident_missing)
    # Should handle gracefully, maybe similarity engine throws error which is caught
    assert len(res_missing["similarity"]["top_cases"]) == 0
    
    # 11. Invalid sensor value
    incident_invalid = incident_normal.copy()
    incident_invalid["RPM"] = "INVALID"
    res_invalid = aggregator.aggregate(incident_invalid)
    assert len(res_invalid["similarity"]["top_cases"]) == 0
    
    print("All Evidence Aggregator tests passed!")
    
    print("\n--- DEMO TEST ---")
    import json
    # Print the aggregated result for the realistic incident with P0403
    print(json.dumps(res_normal, indent=2, default=str))

if __name__ == "__main__":
    test_evidence_aggregator()
