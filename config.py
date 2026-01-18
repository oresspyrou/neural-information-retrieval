import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "IR2025")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Αρχεία Εισόδου (Input)
DOCUMENTS_PATH = os.path.join(RAW_DATA_DIR, "documents.csv")
QUERIES_PATH = os.path.join(RAW_DATA_DIR, "queries.csv")
QRELS_PATH = os.path.join(RAW_DATA_DIR, "qrels.txt")


DOC_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "docs_embeddings.pkl")
QUERY_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "queries_embeddings.pkl") 
MODEL_NAME = 'all-MiniLM-L6-v2' 

BATCH_SIZE = 32

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

DEVICE = get_device()

TOP_K_VALUES = [20, 30, 50]

LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "pipeline.log")