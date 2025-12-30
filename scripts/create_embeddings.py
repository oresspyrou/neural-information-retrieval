import sys
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

# Προσθέτουμε το root directory στο path για να βλέπει το config.py
# (Αυτό χρειάζεται επειδή το script είναι μέσα στο φάκελο scripts/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # Τώρα κάνουμε import το αρχείο που φτιάξαμε!

def create_embeddings():
    # 1. Έλεγχος αρχείων
    if not os.path.exists(config.DOCUMENTS_PATH):
        print(f"❌ Error: Το αρχείο {config.DOCUMENTS_PATH} δεν βρέθηκε.")
        return

    print("🚀 Ξεκινάει η διαδικασία δημιουργίας Embeddings...")
    print(f"⚙️  Model: {config.MODEL_NAME}")
    print(f"⚙️  Device: {config.DEVICE}")

    # 2. Φόρτωση δεδομένων
    print(f"📖 Φόρτωση κειμένων από: {config.DOCUMENTS_PATH}")
    df = pd.read_csv(config.DOCUMENTS_PATH)
    
    doc_ids = df['ID'].tolist()
    texts = df['Text'].tolist()
    
    print(f"✅ Φορτώθηκαν {len(texts)} έγγραφα.")

    # 3. Φόρτωση Μοντέλου
    # Χρησιμοποιούμε τις μεταβλητές από το config
    model = SentenceTransformer(config.MODEL_NAME, device=config.DEVICE)

    # 4. Δημιουργία Embeddings
    print("cw️  Encoding documents...")
    embeddings = model.encode(
        texts, 
        show_progress_bar=True, 
        batch_size=config.BATCH_SIZE
    )

    # 5. Αποθήκευση
    print("💾 Αποθήκευση των embeddings...")
    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
    
    with open(config.DOC_EMBEDDINGS_PATH, "wb") as f:
        pickle.dump({'ids': doc_ids, 'embeddings': embeddings}, f)

    print(f"✅ Επιτυχία! Αποθηκεύτηκαν στο: {config.DOC_EMBEDDINGS_PATH}")
    print(f"📊 Σχήμα: {embeddings.shape}")

if __name__ == "__main__":
    create_embeddings()