# Neural Information Retrieval System

A modular and scalable Dense Retrieval pipeline designed for semantic document search. By leveraging neural embeddings and vector similarity search, this system maps textual data into a continuous vector space to retrieve documents based on semantic meaning rather than simple keyword matching.

## Features & Architecture

The package is strictly divided into distinct components to enforce a clean separation of concerns:

- **Data Preprocessing:** Standardizes and cleans raw text input to prepare it for vector generation (`preprocessing.py`).
- **Embedding Generation:** Transforms processed text documents and incoming user queries into high-dimensional semantic vectors using pre-trained neural models (`create_embeddings.py`).
- **Dense Vector Search:** Utilizes **FAISS (Facebook AI Similarity Search)** to index document embeddings and perform high-speed K-Nearest Neighbors (K-NN) similarity lookups (`faiss_search.py`).
- **Validation & Quality Control:** Ensures integrity of inputs and handles potential structure discrepancies safely (`validator.py`).
- **Standardized Evaluation:** Integrates with standard **TREC evaluation workflows** (`trec_eval.py`) to systematically measure retrieval metrics (e.g., MAP, NDCG, Precision@K).

---

## Project Structure

```text
├── logs/
│   └── pipeline.log         # Execution and tracking logs
├── scripts/
│   ├── __init__.py          # Scripts package indicator
│   ├── preprocessing.py     # Text cleaning and preprocessing module
│   ├── create_embeddings.py # Neural embedding generation script
│   ├── faiss_search.py      # Similarity indexing and search coordination via FAISS
│   ├── validator.py         # Input file validation and checks
│   ├── trec_eval.py         # IR metrics evaluation (MAP, NDCG)
│   ├── logger_setup.py      # System logging configuration
│   └── utils.py             # Shared helper functions
├── __init__.py              # Root package initialization
├── config.py                # Centralized hyperparameters and model configurations
└── requirements.txt         # Project dependencies
