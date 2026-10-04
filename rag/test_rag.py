import os
from retriever import Retriever

def test_rag():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    index_path = os.path.join(base_dir, "ml_engine", "rag", "faiss.index")
    meta_path = os.path.join(base_dir, "ml_engine", "rag", "metadata.pkl")
    
    retriever = Retriever(index_path, meta_path)
    retriever.load()
    
    # 7. Metadata-to-FAISS consistency
    assert len(retriever.metadata) > 0
    assert retriever.index.ntotal == len(retriever.metadata)
    
    # 1. Exact DTC query
    res_exact = retriever.retrieve("P0403", top_k=3)
    assert res_exact[0]["DTC_code"] == "P0403"
    assert res_exact[0]["similarity_score"] == "EXACT_MATCH"
    
    # 2. Semantic query
    res_semantic = retriever.retrieve("EGR control circuit problem", top_k=5)
    # P0403 is "Exhaust Gas Recirculation Control Circuit"
    codes_semantic = [r["DTC_code"] for r in res_semantic]
    assert "P0403" in codes_semantic
    
    # 3. Symptom-oriented query
    res_symptom = retriever.retrieve("EGR issue causing poor engine performance", top_k=5)
    assert len(res_symptom) > 0
    
    # 4. Unknown DTC
    res_unknown = retriever.retrieve("P9999", top_k=3)
    # Should fall back to semantic search and return something, but no EXACT_MATCH
    assert len(res_unknown) > 0
    assert res_unknown[0]["similarity_score"] != "EXACT_MATCH"
    
    # 5. Empty/invalid query
    try:
        retriever.retrieve("", top_k=3)
        assert False, "Should raise ValueError for empty query"
    except ValueError:
        pass
        
    # 6. Top-K behavior
    res_top1 = retriever.retrieve("engine", top_k=1)
    assert len(res_top1) == 1
    res_top10 = retriever.retrieve("engine", top_k=10)
    assert len(res_top10) == 10
    
    # 8. Reproducibility
    res_rep1 = retriever.retrieve("fuel pressure", top_k=3)
    res_rep2 = retriever.retrieve("fuel pressure", top_k=3)
    assert [r["DTC_code"] for r in res_rep1] == [r["DTC_code"] for r in res_rep2]

    print("All RAG tests passed!")
    
    print("\n--- DEMO TEST ---")
    
    print("\nQuery: 'P0403'")
    demo1 = retriever.retrieve("P0403", top_k=2)
    for i, r in enumerate(demo1):
        print(f"Rank {i+1} | DTC: {r['DTC_code']} | Score: {r['similarity_score']} | Title: {r['title']}")
        print(f"  Causes: {r['common_causes']}")
        print(f"  Repair: {r['repair_information']}")
        
    print("\nQuery: 'EGR control circuit problem'")
    demo2 = retriever.retrieve("EGR control circuit problem", top_k=2)
    for i, r in enumerate(demo2):
        print(f"Rank {i+1} | DTC: {r['DTC_code']} | Score: {r['similarity_score']} | Title: {r['title']}")
        print(f"  Causes: {r['common_causes']}")
        print(f"  Repair: {r['repair_information']}")

if __name__ == "__main__":
    test_rag()
