# OBDex RAG Knowledge Retrieval

## Overview
This Retrieval-Augmented Generation (RAG) component provides structured knowledge context about Diagnostic Trouble Codes (DTCs). It loads generic DTC knowledge from the `OBDex` repository, creates a reproducible FAISS vector index, and retrieves relevant contextual information based on semantic or exact-DTC queries.

**Important Disclaimer**: This RAG system is a *knowledge retrieval* system. Retrieving a specific DTC or common cause does **not** establish a confirmed diagnosis or causality for the current vehicle incident. It simply returns reference material to assist investigation.

## Data Source & Parsing
- **Source**: `OBDex-main/data/generic/*.yaml` files.
- **Parsing**: The builder parses each entry to extract code, category, title, description, affected components, common causes, repair information, and flags.
- **Document Construction**: A structured logical document is built from these fields to ensure the semantic embedding captures the full context of the fault without fabricating any missing data.

## Embeddings & Indexing
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (Embedding Dimension: 384). Chosen for its lightweight efficiency and strong semantic matching capabilities.
- **Index**: `FAISS IndexFlatL2`. Computes L2 distances for exact nearest-neighbor search.
- **Metadata**: Stored in `metadata.pkl`, mapping the FAISS vector IDs back to the structured knowledge payload.

## Retrieval Strategy
The `Retriever` implements a dual-layer approach:
1. **Exact-DTC Lookup**: If the query contains a valid DTC format (e.g., `P0403`), the system deterministicically performs a direct metadata lookup first. This prevents semantic similarity from accidentally ranking a different code higher just because the words are similar. Exact matches are given a synthetic `EXACT_MATCH` score.
2. **Semantic Retrieval**: For symptom-oriented queries (e.g., "EGR control circuit problem"), the query is embedded using `all-MiniLM-L6-v2` and nearest neighbors are retrieved from FAISS. The L2 distance is converted to a bounded similarity score (`1 / (1 + distance)`).

## Limitations
- **No Generative Answers**: The system retrieves documents; it does not synthesize new paragraphs. Generation is left to the downstream investigation hypothesis layer.
- **Dependency on OBDex**: The quality of retrieval is entirely dependent on the quality and comprehensiveness of the OBDex dataset. Missing or sparsely populated DTCs will retrieve poorly.
- **No Contextual History**: The retriever relies purely on the query text. It does not natively fuse the query with numerical telemetry (that is the role of the similarity/anomaly engines).
