import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import re

class Retriever:
    def __init__(self, index_path, meta_path):
        self.index_path = index_path
        self.meta_path = meta_path
        
        self.model_name = 'all-MiniLM-L6-v2'
        
        self.index = None
        self.metadata = None
        self.model = None
        
        # Fast lookup mapping for exact DTC lookup
        self.code_to_id = {}

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("FAISS index or metadata not found. Run builder first.")
            
        self.index = faiss.read_index(self.index_path)
        
        with open(self.meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
            
        self.model = SentenceTransformer(self.model_name)
        
        for doc_id, meta in self.metadata.items():
            self.code_to_id[meta["code"].upper()] = doc_id

    def extract_dtcs(self, query):
        """Extracts potential DTC codes from the query string."""
        matches = re.findall(r'[PBUC][0-3]\d{3}', query.upper())
        return list(set(matches))

    def retrieve(self, query, top_k=5):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
            
        if self.index is None:
            self.load()
            
        results = []
        retrieved_codes = set()
        
        # 1. Exact DTC Match Layer
        extracted_codes = self.extract_dtcs(query)
        for code in extracted_codes:
            if code in self.code_to_id:
                doc_id = self.code_to_id[code]
                meta = self.metadata[doc_id]
                
                # Assign a synthetic perfect score for exact match (-1.0 distance or 1.0 sim)
                res = {
                    "DTC_code": meta["code"],
                    "similarity_score": "EXACT_MATCH",
                    "title": meta["title"],
                    "description": meta["description"],
                    "affected_components": meta["affected_components"],
                    "common_causes": meta["common_causes"],
                    "repair_information": meta["repair_information"],
                    "source_file": meta["source_file"]
                }
                results.append(res)
                retrieved_codes.add(code)
                
        # 2. Semantic Similarity Layer
        # Compute embedding for the query
        query_emb = self.model.encode([query])
        query_emb = np.array(query_emb).astype('float32')
        
        # Search FAISS
        # We fetch extra to account for potential exact match duplicates
        distances, indices = self.index.search(query_emb, top_k + len(results))
        
        for i, doc_id in enumerate(indices[0]):
            if doc_id == -1: # FAISS returns -1 if there are not enough results
                continue
                
            meta = self.metadata[doc_id]
            code = meta["code"]
            
            if code in retrieved_codes:
                continue
                
            # L2 distance. Smaller is more similar.
            # Convert to a similarity score bounded 0-1
            l2_dist = distances[0][i]
            sim_score = 1.0 / (1.0 + float(l2_dist))
            
            res = {
                "DTC_code": code,
                "similarity_score": round(sim_score, 4),
                "title": meta["title"],
                "description": meta["description"],
                "affected_components": meta["affected_components"],
                "common_causes": meta["common_causes"],
                "repair_information": meta["repair_information"],
                "source_file": meta["source_file"]
            }
            results.append(res)
            retrieved_codes.add(code)
            
            if len(results) >= top_k:
                break
                
        # Ensure we don't return more than top_k
        return results[:top_k]

if __name__ == "__main__":
    pass
