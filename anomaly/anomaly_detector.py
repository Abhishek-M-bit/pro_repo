import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, model_dir=None, scaler_path=None):
        self.model_dir = model_dir
        self.scaler_path = scaler_path
        
        self.model_path = os.path.join(model_dir, 'anomaly_model.joblib') if model_dir else None
        
        self.sensor_cols = [
            "LOAD_PCT", "ECT", "MAP", "RPM", "VSS", "IAT", 
            "MAF", "FRP", "BARO", "VPWR", "AAT"
        ]
        
        self.model = None
        self.scaler = None
        
        if self.model_path and os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            
        if self.scaler_path and os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)

    def train(self, data_path):
        """
        Trains the Isolation Forest model and saves it.
        We evaluate it against the dataset conceptually by using standard defaults
        but choosing contamination conservatively.
        """
        if not self.scaler:
            raise ValueError("Scaler must be loaded before training")
            
        print("Loading training data for anomaly detection...")
        df = pd.read_csv(data_path)
        
        # Data is already scaled in train_clean.csv, but to be absolutely sure and maintain
        # the abstraction, let's just use the scaled values directly from the CSV since it was scaled.
        # Wait, in preprocessing we scaled train_clean.csv using the scaler! 
        # So train_clean.csv has ALREADY scaled sensor features.
        X_train = df[self.sensor_cols].values
        
        # Isolation Forest
        print("Training Isolation Forest...")
        self.model = IsolationForest(
            n_estimators=100, 
            contamination=0.01, # Assume 1% of the cleaned data might be highly anomalous outliers
            random_state=42
        )
        
        self.model.fit(X_train)
        
        if self.model_path:
            joblib.dump(self.model, self.model_path)
            print(f"Model saved to {self.model_path}")
            
        # We also want to compute the training feature distributions for explainability.
        # We will store the 5th and 95th percentiles of scaled features to identify 'deviations'.
        self.feature_stats = {
            "p5": np.percentile(X_train, 5, axis=0),
            "p95": np.percentile(X_train, 95, axis=0)
        }
        if self.model_dir:
            joblib.dump(self.feature_stats, os.path.join(self.model_dir, 'feature_stats.joblib'))

    def load_feature_stats(self):
        if not hasattr(self, 'feature_stats'):
            stats_path = os.path.join(self.model_dir, 'feature_stats.joblib')
            if os.path.exists(stats_path):
                self.feature_stats = joblib.load(stats_path)
            else:
                self.feature_stats = None

    def _validate_query(self, query):
        for col in self.sensor_cols:
            if col not in query:
                raise ValueError(f"Missing required sensor field: {col}")
            try:
                query[col] = float(query[col])
            except ValueError:
                raise ValueError(f"Sensor value {col} must be numeric.")

    def detect(self, query):
        self._validate_query(query)
        self.load_feature_stats()
        
        if not self.model or not self.scaler:
            raise ValueError("Model or Scaler is not loaded.")
            
        # Extract features
        query_sensors = np.array([[query[col] for col in self.sensor_cols]])
        
        # Scale
        query_scaled = self.scaler.transform(query_sensors)[0]
        
        # Score (Isolation forest score_samples returns negative anomaly score, lower is more anomalous)
        # We want to return a positive score where higher = more anomalous
        # IF decision_function gives positive for normal, negative for anomaly.
        iso_score = self.model.decision_function([query_scaled])[0]
        
        # Transform iso_score to a somewhat interpretable anomaly_score: 
        # e.g., anomaly_score = -iso_score. So > 0 is anomaly, < 0 is normal.
        anomaly_score = float(-iso_score)
        
        is_anomaly = bool(self.model.predict([query_scaled])[0] == -1)
        
        # Severity mapping
        if anomaly_score < 0:
            severity = "normal"
        elif anomaly_score < 0.1:
            severity = "moderate"
        else:
            severity = "high"
            
        # Feature contributions (Explainability via percentile deviation)
        contributions = []
        if self.feature_stats:
            p5 = self.feature_stats["p5"]
            p95 = self.feature_stats["p95"]
            
            for i, col in enumerate(self.sensor_cols):
                val = query_scaled[i]
                deviation = 0.0
                if val < p5[i]:
                    deviation = float(p5[i] - val)
                elif val > p95[i]:
                    deviation = float(val - p95[i])
                    
                if deviation > 0:
                    contributions.append({
                        "feature": col,
                        "deviation_magnitude": deviation,
                        "raw_value": float(query[col])
                    })
                    
        # Sort contributions by deviation magnitude
        contributions.sort(key=lambda x: x["deviation_magnitude"], reverse=True)
        
        return {
            "anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "severity": severity,
            "feature_contributions": contributions
        }

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_path = os.path.join(base_dir, "ml_engine", "data", "processed", "train_clean.csv")
    scaler_path = os.path.join(base_dir, "ml_engine", "models", "preprocessor.joblib")
    model_dir = os.path.join(base_dir, "ml_engine", "models")
    
    detector = AnomalyDetector(model_dir=model_dir, scaler_path=scaler_path)
    detector.train(data_path)
