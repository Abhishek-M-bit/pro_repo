# Backend Integration Guide

This document outlines how to integrate the machine learning pipeline into the backend API.

## Requirements & Dependencies
Ensure the environment running the backend API has the same dependencies as the ML engine:
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `faiss-cpu`
- `sentence-transformers`

The ML artifacts (models and indexes) must be fully built and present in `ml_engine/models/` and `ml_engine/rag/`.

## How to Initialize the ML Pipeline
To prevent reloading heavy models (like the SentenceTransformer or the FAISS index) on every single request, the pipeline employs a singleton pattern. You do not need to initialize it yourself; it initializes automatically on the first call.

## How to Call Inference
Import `analyze_incident` from `ml_engine/inference.py`.

```python
from ml_engine.inference import analyze_incident

# Within your API endpoint (e.g., FastAPI POST /analyze)
evidence_package = analyze_incident(request_body)
```

## Input Schema
The `analyze_incident` function accepts a single `dict`.
It must contain exactly these 11 numeric sensors, `Mode` (integer), and `active_dtcs` (list of strings).

```json
{
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
  "active_dtcs": [
    "P0403",
    "P0404"
  ]
}
```

## Output Structure
The function returns a Python dictionary that is fully JSON-serializable.

### Example Response (Success)
```json
{
  "incident": {
    "LOAD_PCT": 26.3,
    "ECT": 169.0,
    "Mode": 0,
    "active_dtcs": ["P0403"]
  },
  "similarity": {
    "top_cases": [ ... list of dicts ... ],
    "top_score": 0.94,
    "summary": "Similarity calculation successful."
  },
  "anomaly": {
    "anomaly": false,
    "anomaly_score": -0.0815,
    "severity": "normal",
    "feature_contributions": []
  },
  "rag": {
    "exact_dtc_matches": [ ... list of dicts ... ],
    "semantic_matches": [ ... list of dicts ... ]
  },
  "service_history": {
    "available": false,
    "reason": "No real service-history dataset is currently available."
  },
  "evidence_summary": [
    "Found 5 historically similar operating states."
  ],
  "investigation_leads": [
    "Investigate systems/components associated with DTC P0403..."
  ]
}
```

### Example Response (Error)
If validation fails, or an internal error occurs that aborts the pipeline, it returns an error dict. (Internal sub-component failures like an empty RAG result are handled gracefully and still return a success wrapper).
```json
{
  "error": "Input Validation Error",
  "message": "Missing required sensor field: RPM"
}
```

## Expected Response Time
Not benchmarked yet. Early estimates suggest ~100ms-500ms per request (after the first cold-start initialization) depending heavily on the CPU available for the SentenceTransformer encoding and FAISS lookup.
