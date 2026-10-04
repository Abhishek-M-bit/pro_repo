# AI Fleet Service Root-Cause Explorer - ML Engine

## 1. Project Purpose
The ML Engine for the AI Fleet Service Root-Cause Explorer assists vehicle service engineers in investigating new vehicle faults. Instead of acting as an automated "black box" diagnostician, it provides empirical, data-driven evidence by finding historically similar operating states, flagging anomalous sensor deviations, and retrieving explicit diagnostic knowledge. 

## 2. Architecture
The system employs a decoupled, multi-component architecture orchestrated by a unified inference pipeline:
```
Current Incident 
      ↓
Input Validation
      ↓
Preprocessing
      ↓
      ├───────────────┐
      ↓               ↓
Similarity       Anomaly
      │               │
      └───────┬───────┘
              ↓
             RAG
              ↓
      Evidence Aggregation
              ↓
       Final Evidence Package
```

## 3. Folder Structure
- `anomaly/`: Contains the Isolation Forest model and tests.
- `data/`: Contains raw (`raw/`) and cleaned (`processed/`) CSV datasets.
- `evidence/`: Contains the deterministic aggregator logic coordinating all models.
- `models/`: Stores the fitted `StandardScaler`, `IsolationForest`, and statistical joblib artifacts.
- `preprocessing/`: Scripts for data deduplication, DTC signature extraction, and scaling.
- `rag/`: Holds the FAISS index, metadata, and sentence-transformers retriever.
- `similarity/`: Handles mathematical distance scoring (Euclidean/Jaccard).
- `inference.py` & `pipeline.py`: The final API boundary.

## 4. Data Sources
- **Telemetry Data**: Derived from `MASTER_TRAIN_X.xlsx` and `MASTER_TEST_X.xlsx`.
- **Knowledge Data**: Derived from the open-source `OBDex` YAML diagnostic repository.

## 5. Preprocessing
Removes extreme data duplication from the training set, extracts categorical DTC lists into deterministic signatures, handles the `P0000` (no fault) indicator, and fits a standard statistical scaler for the numerical sensor data.

## 6. Similarity Engine
Given an incident, the similarity engine uses an inverse Euclidean distance for numerical sensors, Jaccard similarity for DTC signatures, and exact matching for the operational `Mode` to find historically comparable fault conditions.

## 7. Anomaly Detection
Uses an unsupervised **Isolation Forest** trained strictly on historical sensor data to identify conditions that are numerically rare. Percentile deviations explain which sensors drove the anomaly score.

## 8. RAG (Retrieval-Augmented Generation)
Uses a `FAISS` vector index and `sentence-transformers` (`all-MiniLM-L6-v2`) to retrieve structured YAML diagnostic knowledge. It separates deterministic EXACT DTC matches from SEMANTIC matches to prevent hallucination.

## 9. Evidence Aggregation
The aggregator queries all three independent models and constructs a highly structured, JSON-serializable `Evidence Package`. It implements deterministic rules to build human-readable summaries and investigation leads, completely circumventing LLM hallucinations.

## 10. Inference Interface
`inference.py` acts as the single boundary for backend communication. It loads all heavy models (SentenceTransformer, FAISS, Joblib) precisely once into a global singleton, providing rapid inference without retraining.

## 11. Service-History Limitation
The provided datasets are purely operational telemetry and explicitly lack repair logs, confirmed fixes, or customer complaints. Therefore, the aggregator hardcodes `service_history` availability to `False`. The ML engine does **not** fabricate synthetic service history or claim to confirm a root cause.

## 12. Example Input
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
  "active_dtcs": ["P0403"]
}
```

## 13. Example Output
```json
{
  "incident": {...},
  "similarity": {"top_cases": [...], "top_score": 0.94, "summary": "..."},
  "anomaly": {"anomaly": false, "anomaly_score": -0.08, "severity": "normal", "feature_contributions": []},
  "rag": {"exact_dtc_matches": [...], "semantic_matches": []},
  "service_history": {"available": false, "reason": "..."},
  "evidence_summary": ["..."],
  "investigation_leads": ["..."]
}
```

## 14. Backend Integration
Please see `ml_engine/INTEGRATION.md` for exact integration steps, API contracts, and usage examples.
