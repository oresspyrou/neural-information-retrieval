import os
import torch

# --- ΒΑΣΙΚΑ PATHS ---
# Βρίσκουμε το μονοπάτι του φακέλου που βρισκόμαστε τώρα (root του project)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Φάκελοι Δεδομένων
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "IR2025")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Αρχεία Εισόδου (Input)
DOCUMENTS_PATH = os.path.join(RAW_DATA_DIR, "documents.csv")
QUERIES_PATH = os.path.join(RAW_DATA_DIR, "queries.csv")
QRELS_PATH = os.path.join(RAW_DATA_DIR, "qrels.txt")

# Αρχεία Εξόδου (Output Embeddings)
DOC_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "docs_embeddings.pkl")
QUERY_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "queries_embeddings.pkl") # Αν θελήσουμε να τα σώσουμε

# --- ΡΥΘΜΙΣΕΙΣ ΜΟΝΤΕΛΟΥ (MODEL CONFIG) ---
# Το όνομα του μοντέλου στο HuggingFace
# Προτάσεις: 'all-MiniLM-L6-v2' (Γρήγορο/Ελαφρύ) ή 'distilbert-base-uncased'
MODEL_NAME = 'all-MiniLM-L6-v2' 

# Batch Size: Πόσα κείμενα επεξεργάζεται ταυτόχρονα
# Αν έχεις GPU βάλε 32 ή 64. Αν έχεις CPU βάλε 16 ή 32.
BATCH_SIZE = 32

# --- ΡΥΘΜΙΣΕΙΣ DEVICE (CPU/GPU) ---
# Αυτόματα βρίσκει αν έχεις NVIDIA GPU (cuda), Mac M1/M2 (mps) ή CPU
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available(): # Για Mac Users
        return 'mps'
    else:
        return 'cpu'

DEVICE = get_device()

# --- ΡΥΘΜΙΣΕΙΣ ΑΝΑΖΗΤΗΣΗΣ (SEARCH CONFIG) ---
# Τα k για τα οποία θα τρέξουμε αξιολόγηση
TOP_K_VALUES = [20, 30, 50]