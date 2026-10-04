import os
import sys

# Add ml_engine to sys.path to allow imports from other submodules
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ml_engine_dir = os.path.join(base_dir, "ml_engine")
if ml_engine_dir not in sys.path:
    sys.path.append(ml_engine_dir)

from similarity.similarity_engine import SimilarityEngine
from anomaly.anomaly_detector import AnomalyDetector
from rag.retriever import Retriever

class EvidenceAggregator:
    def __init__(self, data_path, scaler_path, anomaly_model_dir, rag_index_path, rag_meta_path):
        self.similarity_engine = SimilarityEngine(data_path, scaler_path)
        self.anomaly_detector = AnomalyDetector(model_dir=anomaly_model_dir, scaler_path=scaler_path)
        self.rag_retriever = Retriever(rag_index_path, rag_meta_path)
        self.rag_retriever.load()

    def aggregate(self, incident, top_k_similar=5, top_k_rag=3):
        # 1. Similarity
        try:
            similar_cases = self.similarity_engine.find_similar(incident, top_k=top_k_similar)
            top_score = similar_cases[0]["similarity_score"] if similar_cases else 0.0
        except Exception as e:
            similar_cases = []
            top_score = 0.0
            print(f"Similarity Engine error: {e}")

        # 2. Anomaly
        try:
            anomaly_result = self.anomaly_detector.detect(incident)
        except Exception as e:
            anomaly_result = {"anomaly": False, "anomaly_score": 0.0, "severity": "unknown", "feature_contributions": [], "error": str(e)}

        # 3. RAG Retriever
        exact_matches = []
        semantic_matches = []
        try:
            active_dtcs = incident.get("active_dtcs", [])
            for dtc in active_dtcs:
                if dtc == "P0000" or dtc == "NONE":
                    continue
                # Retrieve using the DTC code as query
                rag_res = self.rag_retriever.retrieve(dtc, top_k=top_k_rag)
                for res in rag_res:
                    if res["similarity_score"] == "EXACT_MATCH":
                        if res not in exact_matches:
                            exact_matches.append(res)
                    else:
                        if res not in semantic_matches:
                            semantic_matches.append(res)
        except Exception as e:
            print(f"RAG Retriever error: {e}")

        # 4. Generate Evidence Summary (Deterministic rules)
        summary = []
        if similar_cases:
            summary.append(f"Found {len(similar_cases)} historically similar operating states.")
            # Check if any top case has exact same DTC signature
            query_sig = "|".join(sorted(incident.get("active_dtcs", []))) if incident.get("active_dtcs") else "NONE"
            if any(c["dtc_signature"] == query_sig for c in similar_cases):
                summary.append("A historical operating state with the same DTC signature was found.")
                
        if anomaly_result.get("anomaly") is True:
            summary.append("The current operating condition was classified as anomalous.")
            feats = anomaly_result.get("feature_contributions", [])
            if feats:
                summary.append(f"{feats[0]['feature']} was among the features with the largest deviation.")

        if exact_matches:
            dtcs_found = [m["DTC_code"] for m in exact_matches]
            summary.append(f"OBDex contains knowledge associated with DTC(s): {', '.join(dtcs_found)}.")

        # 5. Generate Investigation Leads
        leads = []
        if exact_matches:
            dtcs_found = [m["DTC_code"] for m in exact_matches]
            leads.append(f"Investigate systems/components associated with DTC {', '.join(dtcs_found)} using the retrieved OBDex evidence.")
            
        if similar_cases:
            leads.append("Review the operating conditions of the retrieved historical cases to understand typical steady-state behavior for this profile.")
            
        if anomaly_result.get("anomaly") is True:
            feats = anomaly_result.get("feature_contributions", [])
            if feats:
                leads.append(f"Investigate the sensor parameters contributing most strongly to the anomalous operating state, particularly {feats[0]['feature']}.")
                
        if exact_matches or semantic_matches:
            leads.append("Review the common causes documented in the retrieved DTC knowledge.")

        # Build Package
        package = {
            "incident": incident,
            "similarity": {
                "top_cases": similar_cases,
                "top_score": top_score,
                "summary": "Similarity calculation successful." if similar_cases else "No similar cases found or error."
            },
            "anomaly": anomaly_result,
            "rag": {
                "exact_dtc_matches": exact_matches,
                "semantic_matches": semantic_matches
            },
            "service_history": {
                "available": False,
                "reason": "No real service-history dataset is currently available."
            },
            "evidence_summary": summary,
            "investigation_leads": leads
        }
        
        return package

if __name__ == "__main__":
    pass
