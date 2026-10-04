import pandas as pd
import numpy as np
import joblib
import os
import ast
from sklearn.metrics.pairwise import euclidean_distances

class SimilarityEngine:
    def __init__(self, data_path, scaler_path, weights=None):
        self.data_path = data_path
        self.scaler_path = scaler_path
        
        # Configurable weights
        self.weights = weights if weights else {
            "sensor": 0.5,
            "dtc": 0.3,
            "mode": 0.2
        }
        
        self.sensor_cols = [
            "LOAD_PCT", "ECT", "MAP", "RPM", "VSS", "IAT", 
            "MAF", "FRP", "BARO", "VPWR", "AAT"
        ]
        
        self._load_resources()

    def _load_resources(self):
        # Load scaler
        self.scaler = joblib.load(self.scaler_path)
        
        # Load historical data
        self.history_df = pd.read_csv(self.data_path)
        
        # Ensure active_dtcs is evaluated as list if it's string
        if 'active_dtcs' in self.history_df.columns:
            # We saved it as string representation of list, let's parse safely
            def parse_list(x):
                if isinstance(x, str):
                    try:
                        return ast.literal_eval(x)
                    except:
                        return []
                return []
            self.history_df['active_dtcs_list'] = self.history_df['active_dtcs'].apply(parse_list)
            
        # The sensor data in train_clean.csv is already scaled. 
        # We will keep a copy of unscaled values if possible, but the prompt says 
        # train_clean.csv is the output of preprocessing.
        # Wait, if train_clean.csv has scaled values, we can't easily return original values 
        # unless we inverse transform.
        self.history_scaled_sensors = self.history_df[self.sensor_cols].values
        
    def _validate_query(self, query):
        # Required keys
        for col in self.sensor_cols:
            if col not in query:
                raise ValueError(f"Missing required sensor field: {col}")
            try:
                query[col] = float(query[col])
            except ValueError:
                raise ValueError(f"Sensor value {col} must be numeric.")
                
        if "Mode" not in query:
            raise ValueError("Missing required field: Mode")
        
        if "active_dtcs" not in query:
            raise ValueError("Missing required field: active_dtcs (list of strings)")
            
        if not isinstance(query["active_dtcs"], list):
            raise ValueError("active_dtcs must be a list of strings")

    def _sensor_similarity(self, query_scaled):
        # Calculate euclidean distance
        dists = euclidean_distances([query_scaled], self.history_scaled_sensors)[0]
        # Convert distance to similarity in [0, 1]
        sims = 1.0 / (1.0 + dists)
        return sims

    def _dtc_similarity(self, query_dtcs):
        query_set = set(query_dtcs)
        if not query_set:
            # If query has no faults, then historical rows with no faults get 1.0, else 0.0
            def sim_no_fault(hist_list):
                hist_set = set(hist_list)
                if not hist_set or hist_set == {"P0000"} or hist_set == {"NONE"}:
                    return 1.0
                return 0.0
            return self.history_df['active_dtcs_list'].apply(sim_no_fault).values
            
        def jaccard(hist_list):
            hist_set = set(hist_list)
            if not hist_set:
                return 0.0
            intersection = query_set.intersection(hist_set)
            union = query_set.union(hist_set)
            if not union:
                return 0.0
            return len(intersection) / len(union)
            
        return self.history_df['active_dtcs_list'].apply(jaccard).values

    def _mode_similarity(self, query_mode):
        hist_mode = self.history_df['Mode'].values
        return (hist_mode == query_mode).astype(float)

    def find_similar(self, query, top_k=5):
        self._validate_query(query)
        
        # 1. Preprocess query (scale sensors)
        query_sensors = np.array([[query[col] for col in self.sensor_cols]])
        query_scaled = self.scaler.transform(query_sensors)[0]
        
        # 2. Calculate component similarities
        sensor_sim = self._sensor_similarity(query_scaled)
        dtc_sim = self._dtc_similarity(query["active_dtcs"])
        mode_sim = self._mode_similarity(query["Mode"])
        
        # 3. Combined score
        w = self.weights
        final_scores = (w['sensor'] * sensor_sim) + (w['dtc'] * dtc_sim) + (w['mode'] * mode_sim)
        
        # 4. Top K retrieval
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            hist_row = self.history_df.iloc[idx]
            
            # Inverse transform to get original sensor values for the result
            scaled_vals = hist_row[self.sensor_cols].values.reshape(1, -1)
            orig_vals = self.scaler.inverse_transform(scaled_vals)[0]
            
            orig_sensor_dict = {col: orig_vals[i] for i, col in enumerate(self.sensor_cols)}
            
            res = {
                "historical_index": int(idx),
                "similarity_score": float(final_scores[idx]),
                "sensor_similarity": float(sensor_sim[idx]),
                "dtc_similarity": float(dtc_sim[idx]),
                "mode_similarity": float(mode_sim[idx]),
                "dtc_signature": hist_row.get("dtc_signature", "NONE"),
                "Mode": int(hist_row["Mode"]),
                "original_sensor_values": orig_sensor_dict
            }
            results.append(res)
            
        return results
