import sys
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  

def create_embeddings():

    if not os.path.exists(config.DOCUMENTS_PATH):
        print(f"Error: File {config.DOCUMENTS_PATH} not found.")
        return

    print("Creating embeddings...")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Device: {config.DEVICE}")

    print(f"Loading texts from: {config.DOCUMENTS_PATH}")
    df = pd.read_csv(config.DOCUMENTS_PATH)
    
    doc_ids = df['ID'].tolist()
    texts = df['Text'].tolist()
    print(f"Loaded {len(texts)} documents.")

    model = SentenceTransformer(config.MODEL_NAME, device=config.DEVICE)

    print("Encoding documents...")
    embeddings = model.encode(
        texts, 
        show_progress_bar=True, 
        batch_size=config.BATCH_SIZE
    )

    # 5. Αποθήκευση
    print("Saving embeddings...")
    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
    
    with open(config.DOC_EMBEDDINGS_PATH, "wb") as f:
        pickle.dump({'ids': doc_ids, 'embeddings': embeddings}, f)

    print(f"Saved embeddings to: {config.DOC_EMBEDDINGS_PATH}")
    print(f"Shape: {embeddings.shape}")

if __name__ == "__main__":
    create_embeddings()