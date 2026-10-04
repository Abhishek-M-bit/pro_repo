import os
import pandas as pd
import numpy as np
from similarity_engine import SimilarityEngine
import joblib

def test_similarity_engine():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_path = os.path.join(base_dir, "ml_engine", "data", "processed", "train_clean.csv")
    scaler_path = os.path.join(base_dir, "ml_engine", "models", "preprocessor.joblib")
    
    engine = SimilarityEngine(data_path, scaler_path)
    
    # Extract a real row to test
    df = pd.read_csv(data_path)
    scaler = joblib.load(scaler_path)
    
    # Real query from index 0
    row_0 = df.iloc[0]
    sensor_cols = engine.sensor_cols
    
    # We need to inverse transform the row to get original sensor values for the query
    scaled_vals = row_0[sensor_cols].values.reshape(1, -1)
    orig_vals = scaler.inverse_transform(scaled_vals)[0]
    
    real_query = {col: orig_vals[i] for i, col in enumerate(sensor_cols)}
    real_query["Mode"] = int(row_0["Mode"])
    
    # Parse active DTCs safely
    import ast
    try:
        active = ast.literal_eval(row_0["active_dtcs"]) if isinstance(row_0["active_dtcs"], str) else []
    except:
        active = []
    real_query["active_dtcs"] = active
    
    # Test Demo: Retrieve itself
    res = engine.find_similar(real_query, top_k=5)
    assert len(res) == 5
    top_res = res[0]
    
    # 1. Returned results contain all required fields
    expected_fields = ["historical_index", "similarity_score", "sensor_similarity", 
                       "dtc_similarity", "mode_similarity", "dtc_signature", 
                       "Mode", "original_sensor_values"]
    for f in expected_fields:
        assert f in top_res
        
    # 2. Exact DTC match and Same Mode should yield very high score (querying itself)
    # The first result should be the same row (or an identical duplicate if any survived)
    # Since it's the exact same query, sensor_sim should be ~1.0, dtc_sim ~1.0, mode_sim ~1.0
    assert top_res["sensor_similarity"] > 0.99
    assert top_res["dtc_similarity"] == 1.0
    assert top_res["mode_similarity"] == 1.0
    assert top_res["similarity_score"] > 0.99
    
    # 3. Top-K ordering
    scores = [r["similarity_score"] for r in res]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), "Not sorted by score"
    
    # 4. Partial DTC match
    query_partial = real_query.copy()
    query_partial["active_dtcs"] = [active[0]] if active else ["P9999"]
    res_partial = engine.find_similar(query_partial, top_k=1)
    # DTC sim should be < 1.0 if the top result has multiple
    # We just ensure it runs and computes a score
    assert "dtc_similarity" in res_partial[0]
    
    # 5. No DTC match
    query_no_dtc = real_query.copy()
    query_no_dtc["active_dtcs"] = ["P9999"] # A fake DTC
    res_no = engine.find_similar(query_no_dtc, top_k=1)
    # Find the top result. If it doesn't have P9999, dtc_sim = 0
    if "P9999" not in res_no[0]["dtc_signature"]:
        assert res_no[0]["dtc_similarity"] == 0.0
        
    # 6. Different Mode
    query_diff_mode = real_query.copy()
    query_diff_mode["Mode"] = 99 # Fake mode
    res_mode = engine.find_similar(query_diff_mode, top_k=1)
    if res_mode[0]["Mode"] != 99:
        assert res_mode[0]["mode_similarity"] == 0.0
        
    # 7. Invalid query handling
    try:
        bad_query = real_query.copy()
        del bad_query["RPM"]
        engine.find_similar(bad_query)
        assert False, "Should have raised ValueError for missing RPM"
    except ValueError:
        pass
        
    try:
        bad_query = real_query.copy()
        bad_query["RPM"] = "NOT_A_NUMBER"
        engine.find_similar(bad_query)
        assert False, "Should have raised ValueError for non-numeric RPM"
    except ValueError:
        pass
        
    # 8. Query preprocessing uses the saved scaler
    # Verified by the fact that the self-query achieved > 0.99 similarity
    
    print("All Similarity Engine tests passed!")
    
    print("\n--- DEMO TEST ---")
    print("Query:", real_query)
    print("\nTop 5 Results:")
    for i, r in enumerate(res):
        print(f"Rank {i+1}: Score={r['similarity_score']:.4f} "
              f"(Sensor={r['sensor_similarity']:.4f}, DTC={r['dtc_similarity']:.4f}, Mode={r['mode_similarity']:.4f}) "
              f"| Signature: {r['dtc_signature']} | Mode: {r['Mode']}")

if __name__ == "__main__":
    test_similarity_engine()
