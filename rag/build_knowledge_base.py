import os
import yaml
import glob
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class KnowledgeBaseBuilder:
    def __init__(self, obdex_dir, output_dir):
        self.obdex_dir = obdex_dir
        self.output_dir = output_dir
        self.model_name = 'all-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.model_name)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _parse_yaml_files(self):
        yaml_files = glob.glob(os.path.join(self.obdex_dir, "data", "generic", "*.yaml"))
        documents = []
        metadata = {}
        
        doc_id = 0
        for yf in yaml_files:
            file_name = os.path.basename(yf)
            with open(yf, 'r', encoding='utf-8') as f:
                entries = yaml.safe_load(f)
                
            if not entries:
                continue
                
            for entry in entries:
                code = entry.get('code')
                if not code:
                    continue
                    
                # Language extraction (prefer EN)
                def get_lang(field, default=""):
                    if not field: return default
                    if isinstance(field, dict):
                        return field.get('en', list(field.values())[0] if field else default)
                    return str(field)
                
                title = get_lang(entry.get('title'))
                desc = get_lang(entry.get('description'))
                category = entry.get('category', '')
                
                components = entry.get('affected_components', [])
                comps_str = ", ".join(components) if components else "None"
                
                causes = entry.get('common_causes', [])
                causes_list = [get_lang(c.get('label')) for c in causes if 'label' in c]
                causes_str = "; ".join(causes_list) if causes_list else "None"
                
                repair = entry.get('repair', {})
                repair_str = (
                    f"Difficulty: {repair.get('difficulty', 'Unknown')}, "
                    f"DIY Possible: {repair.get('diy_possible', 'Unknown')}, "
                    f"Cost EUR: {repair.get('estimated_cost_eur', 'Unknown')}, "
                    f"Hours: {repair.get('estimated_hours', 'Unknown')}"
                )
                
                flags = entry.get('flags', {})
                flags_str = ", ".join([f"{k}: {v}" for k, v in flags.items()]) if flags else "None"
                
                text_doc = (
                    f"DTC: {code}\n"
                    f"Category: {category}\n"
                    f"Title: {title}\n"
                    f"Description: {desc}\n"
                    f"Affected Components: {comps_str}\n"
                    f"Common Causes: {causes_str}\n"
                    f"Repair Information: {repair_str}\n"
                    f"Flags: {flags_str}"
                )
                
                documents.append(text_doc)
                
                metadata[doc_id] = {
                    "code": code,
                    "title": title,
                    "description": desc,
                    "affected_components": components,
                    "common_causes": causes_list,
                    "repair_information": repair,
                    "flags": flags,
                    "source_file": file_name,
                    "text_doc": text_doc
                }
                
                doc_id += 1
                
        return documents, metadata
        
    def build(self):
        print("Parsing YAML files...")
        documents, metadata = self._parse_yaml_files()
        
        print(f"Parsed {len(documents)} DTC entries. Encoding...")
        embeddings = self.model.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        dim = embeddings.shape[1]
        print(f"Building FAISS Index (L2), dimension: {dim}...")
        
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        
        index_path = os.path.join(self.output_dir, "faiss.index")
        faiss.write_index(index, index_path)
        
        meta_path = os.path.join(self.output_dir, "metadata.pkl")
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        print("FAISS index and metadata successfully created.")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    obdex_dir = os.path.join(base_dir, "OBDex-main")
    output_dir = os.path.join(base_dir, "ml_engine", "rag")
    
    builder = KnowledgeBaseBuilder(obdex_dir, output_dir)
    builder.build()
