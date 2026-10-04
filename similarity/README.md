# Structured Similarity Engine

## Overview
The Structured Similarity Engine is designed to identify historical vehicle operating conditions that are most similar to a current queried incident. It leverages normalized sensor data, active Diagnostic Trouble Codes (DTCs), and the vehicle's operating mode to calculate a deterministic and explainable similarity score.

## Features Used
1. **Sensor Features**: `LOAD_PCT`, `ECT`, `MAP`, `RPM`, `VSS`, `IAT`, `MAF`, `FRP`, `BARO`, `VPWR`, `AAT`. These continuous numerical values provide the physiological state of the engine.
2. **DTC Features**: The `active_dtcs` list. Indicates active faults.
3. **Categorical Features**: `Mode`. Captures the driving condition (e.g., Idle, Drive).

## Similarity Methodologies

### 1. Numerical Similarity
Calculated using the **Euclidean Distance** between the query's normalized sensor vector and the historical normalized sensor vectors. To convert distance to a similarity score bounded between 0 and 1, the transformation `1 / (1 + distance)` is applied. 
- *Why this approach?* It guarantees that identical sensor readings yield a 1.0 similarity, smoothly decaying towards 0 as the condition diverges.

### 2. DTC Similarity
Calculated using the **Jaccard Index** (`Intersection / Union`) over the sets of active DTCs.
- An exact match yields 1.0.
- A partial overlap (e.g., query has P0403, history has P0403 and P0404) yields a fractional score.
- No overlap yields 0.0.
- Queries indicating no active faults specifically target historical rows with no active faults.

### 3. Mode Similarity
Calculated using **Exact Match**. If the query Mode exactly equals the historical Mode, the score is 1.0; otherwise, it is 0.0. It is not treated as a continuous variable.

## Combined Score
The final similarity score is a transparent, weighted sum of the three components:
```python
final_score = (sensor_weight * sensor_similarity) + 
              (dtc_weight * dtc_similarity) + 
              (mode_weight * mode_similarity)
```

### Initial Weights
- **Sensor Weight**: 0.5
- **DTC Weight**: 0.3
- **Mode Weight**: 0.2

*Note: These are prototype weights designed for initial interpretability and balance. They are fully configurable and can be empirically tuned later.*

## Top-K Retrieval
The engine returns the `top_k` (default 5) historical cases ranked by the `final_score`. It provides full explainability by returning the breakdown of individual similarities (`sensor_similarity`, `dtc_similarity`, `mode_similarity`), ensuring downstream systems can interpret the reasoning.

## Limitations
1. **Not a Causal Model**: A high similarity score means the vehicle operating states are historically similar; it **does not** confirm a causal relationship or root cause for a failure.
2. **Linear Scalability**: Currently computes distances across all 27,820 deduplicated rows. While performant enough for this volume, massive scale-ups will require a vector database like FAISS.
3. **Weight Calibration**: The fixed weights represent a heuristic priority (sensors > dtcs > mode) rather than an optimal learned distribution.
