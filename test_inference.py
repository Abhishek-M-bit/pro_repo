import os
import json
from inference import analyze_incident

def test_inference_interface():
    # 1. Valid realistic incident / 3. Multiple DTCs / 8. End-to-end pipeline
    demo_incident = {
        "LOAD_PCT": 26.3,
        "ECT": 169.0,
        "MAP": 14.4,
        "RPM": 790.0,
        "VSS": 0.0,
        "IAT": 106.0,
        "MAF": 0.02,
        "FRP": 4507.5,
        "BARO": 14.2,
        "VPWR": 13.64,
        "AAT": 126.0,
        "Mode": 0,
        "active_dtcs": ["P0403", "P0404"]
    }
    
    res1 = analyze_incident(demo_incident)
    assert "error" not in res1, f"Failed with {res1.get('message')}"
    assert "incident" in res1
    assert "similarity" in res1
    assert "rag" in res1
    
    # 9. JSON serialization
    try:
        json_str = json.dumps(res1)
        assert len(json_str) > 0
    except TypeError:
        assert False, "Output is not JSON serializable"
        
    # 10. Repeatability (models are not re-loaded, scores should be identical)
    res2 = analyze_incident(demo_incident)
    assert res1["anomaly"]["anomaly_score"] == res2["anomaly"]["anomaly_score"]
    
    # 2. Single DTC
    inc_single = demo_incident.copy()
    inc_single["active_dtcs"] = ["P0403"]
    res_single = analyze_incident(inc_single)
    assert "error" not in res_single
    
    # 4. No DTC
    inc_none = demo_incident.copy()
    inc_none["active_dtcs"] = []
    res_none = analyze_incident(inc_none)
    assert "error" not in res_none
    
    # 5. Invalid sensor
    inc_invalid_sensor = demo_incident.copy()
    inc_invalid_sensor["RPM"] = "INVALID"
    res_invalid_sensor = analyze_incident(inc_invalid_sensor)
    assert "error" in res_invalid_sensor
    assert "must be numeric" in res_invalid_sensor["message"]
    
    # 6. Missing sensor
    inc_missing_sensor = demo_incident.copy()
    del inc_missing_sensor["ECT"]
    res_missing_sensor = analyze_incident(inc_missing_sensor)
    assert "error" in res_missing_sensor
    assert "Missing required sensor field" in res_missing_sensor["message"]
    
    # 7. Invalid Mode
    inc_invalid_mode = demo_incident.copy()
    inc_invalid_mode["Mode"] = "INVALID_MODE"
    res_invalid_mode = analyze_incident(inc_invalid_mode)
    assert "error" in res_invalid_mode
    assert "Mode must be an integer" in res_invalid_mode["message"]

    print("All Inference Interface tests passed!")

if __name__ == "__main__":
    test_inference_interface()
