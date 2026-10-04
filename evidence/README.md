# Evidence Aggregation

## Purpose
The Evidence Aggregation layer serves as the orchestrator for the AI Fleet Service Root-Cause Explorer. It takes a current vehicle incident and systematically queries the three independent foundational models:
1. **Structured Similarity Engine**: To find historically similar operating conditions.
2. **Anomaly Detector**: To evaluate numerical rarity and deviations.
3. **OBDex RAG Retriever**: To fetch standardized DTC knowledge and repair information.

It then formats these independent outputs into a single, structured JSON payload suitable for downstream ingestion by a backend API or a frontend UI.

## Subsystem Orchestration
The aggregator explicitly decouples the components so that a failure in one (e.g., the RAG index being unavailable) does not crash the entire request.
- If a query lacks DTCs, RAG safely returns empty.
- If sensor readings are malformed, Similarity and Anomaly handle the validation errors gracefully while still returning the error context in the payload.

## Inputs
A JSON/dictionary object representing the vehicle state:
```json
{
    "LOAD_PCT": 26.3,
    "ECT": 169.0,
    ...
    "Mode": 0,
    "active_dtcs": ["P0403", "P0404"]
}
```

## Outputs
A comprehensive `Evidence Package` mapping exactly to the schema required for investigation:
- `incident`: The original query.
- `similarity`: The top historical cases and similarity score.
- `anomaly`: The boolean flag, severity, and feature contributions.
- `rag`: Separated lists for exact DTC matches vs semantic matches.
- `service_history`: A hardcoded unavailability block.
- `evidence_summary`: Human-readable string summaries deterministically generated from the data.
- `investigation_leads`: Actionable but conservative next steps.

## Evidence Rules
The aggregation layer generates text deterministically. It does **not** employ an LLM to hallucinate interpretations. 
- Example: If the anomaly detector returns `True`, the aggregator appends the string `"The current operating condition was classified as anomalous."`
- Example: If the anomaly detector flags `ECT` as the highest deviating feature, the aggregator appends `"Investigate the sensor parameters contributing most strongly to the anomalous operating state, particularly ECT."`

## Limitations & Disclaimer
1. **No Confirmed Diagnosis**: The output is an `Evidence Package`, not a diagnosis. The aggregator explicitly avoids stating "The root cause is X."
2. **Service History Limitation**: Because the available dataset lacks real-world repair logs, customer complaints, and downtime metrics, the `service_history.available` flag is permanently hardcoded to `False`. We explicitly document that this data is missing. The aggregator does not fabricate synthetic repair logs.
