import os
import sys

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ml_engine_dir = os.path.join(base_dir, "ml_engine")
if ml_engine_dir not in sys.path:
    sys.path.append(ml_engine_dir)

from evidence.evidence_aggregator import EvidenceAggregator

class FleetInvestigationPipeline:
    def __init__(self, data_path=None, scaler_path=None, anomaly_model_dir=None, rag_index_path=None, rag_meta_path=None):
        if data_path is None:
            data_path = os.path.join(ml_engine_dir, "data", "processed", "train_clean.csv")
        if scaler_path is None:
            scaler_path = os.path.join(ml_engine_dir, "models", "preprocessor.joblib")
        if anomaly_model_dir is None:
            anomaly_model_dir = os.path.join(ml_engine_dir, "models")
        if rag_index_path is None:
            rag_index_path = os.path.join(ml_engine_dir, "rag", "faiss.index")
        if rag_meta_path is None:
            rag_meta_path = os.path.join(ml_engine_dir, "rag", "metadata.pkl")
            
        self.aggregator = EvidenceAggregator(
            data_path=data_path,
            scaler_path=scaler_path,
            anomaly_model_dir=anomaly_model_dir,
            rag_index_path=rag_index_path,
            rag_meta_path=rag_meta_path
        )
        
        self.required_sensors = [
            "LOAD_PCT", "ECT", "MAP", "RPM", "VSS", "IAT", 
            "MAF", "FRP", "BARO", "VPWR", "AAT"
        ]

    def _validate_input(self, incident):
        if not isinstance(incident, dict):
            raise ValueError("Incident must be a dictionary.")
            
        for col in self.required_sensors:
            if col not in incident:
                raise ValueError(f"Missing required sensor field: {col}")
            try:
                incident[col] = float(incident[col])
            except ValueError:
                raise ValueError(f"Sensor value {col} must be numeric.")
                
        if "Mode" not in incident:
            raise ValueError("Missing required field: Mode")
            
        try:
            incident["Mode"] = int(incident["Mode"])
        except ValueError:
            raise ValueError("Mode must be an integer.")
            
        if "active_dtcs" not in incident:
            incident["active_dtcs"] = []
            
        if not isinstance(incident["active_dtcs"], list):
            raise ValueError("active_dtcs must be a list of strings.")

    def analyze(self, incident):
        self._validate_input(incident)
        
        # The Evidence Aggregator orchestrates similarity, anomaly, and RAG.
        # It handles partial failures internally and guarantees the contract.
        package = self.aggregator.aggregate(incident)
        
        # Ensure it is purely python native objects for JSON serialization
        import json
        try:
            # We round trip through JSON to enforce strict serialization
            # and catch any numpy types sneaking through
            package_json = json.dumps(package, default=str)
            package = json.loads(package_json)
        except Exception as e:
            package["serialization_error"] = str(e)
            
        return package
