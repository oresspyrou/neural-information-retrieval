import sys
import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scripts.logger_setup import setup_logger
from scripts.validator import validate_input_file, validate_csv_columns, ensure_directory_exists
from scripts.utils import set_seed
from scripts.preprocessing import clean_text  

try:
    logger = setup_logger()
except RuntimeError as e:
    print(f"CRITICAL: Logger setup failed: {e}")
    sys.exit(1)

def load_embeddings(filepath: str) -> Tuple[List[str], np.ndarray]:
    """
    Loads embeddings and IDs from a pickle file.
    Returns: (List of IDs, Numpy Array of Embeddings)
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data['ids'], data['embeddings']

def run_search() -> None:
    set_seed(42)
    logger.info("Starting Semantic Search pipeline...")

    logger.info("Running validation checks...")
    try:
        validate_input_file(config.QUERIES_PATH)
        validate_csv_columns(config.QUERIES_PATH, required_columns=['ID', 'Text'])
        validate_input_file(config.DOC_EMBEDDINGS_PATH)
        ensure_directory_exists(config.RESULTS_DIR)
        
    except Exception as e:
        logger.error(f"Validation Error: {e}")
        return

    try:
        logger.info(f"Loading document vectors from: {config.DOC_EMBEDDINGS_PATH}")
        doc_ids, doc_embeddings = load_embeddings(config.DOC_EMBEDDINGS_PATH)
        logger.info(f"Loaded {len(doc_ids)} document vectors.")

        logger.info(f"Loading queries from: {config.QUERIES_PATH}")
        queries_df: pd.DataFrame = pd.read_csv(config.QUERIES_PATH)
        
        queries_df['Text'] = queries_df['Text'].apply(clean_text)
        
        query_ids: List[str] = queries_df['ID'].tolist()
        query_texts: List[str] = queries_df['Text'].tolist()
        logger.info(f"Loaded and preprocessed {len(query_texts)} queries.")

    except Exception as e:
        logger.error(f"Data Loading Error: {e}")
        return

    try:
        logger.info(f"Loading model: {config.MODEL_NAME}...")
        model: SentenceTransformer = SentenceTransformer(config.MODEL_NAME, device=config.DEVICE)

        logger.info("Encoding queries to vectors...")
        query_embeddings: np.ndarray = model.encode(
            query_texts, 
            show_progress_bar=True, 
            batch_size=config.BATCH_SIZE
        )

        logger.info("Building FAISS index...")
        
        doc_embeddings = doc_embeddings.astype('float32')
        query_embeddings = query_embeddings.astype('float32')
        
        # Dimensions of vectors
        dimension: int = doc_embeddings.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(doc_embeddings)
        
        logger.info(f"Searching for top {max(config.TOP_K_VALUES)} results per query...")
        
        k: int = max(config.TOP_K_VALUES)
        D, I = index.search(query_embeddings, k)

        logger.info("Formatting results...")
        results: Dict[str, List[Tuple[str, float]]] = {}
        
        for q_idx, neighbors in enumerate(I):
            q_id: str = str(query_ids[q_idx]) 
            
            retrieved_docs: List[Tuple[str, float]] = []
            for i, doc_idx in enumerate(neighbors):
                score: float = float(D[q_idx][i])     
                real_doc_id: str = str(doc_ids[doc_idx])
                
                retrieved_docs.append((real_doc_id, score))
            
            results[q_id] = retrieved_docs

        output_path: str = os.path.join(config.RESULTS_DIR, "search_results.pkl")
        logger.info(f"Saving results to: {output_path}")
        
        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        logger.info("Search pipeline completed successfully!")

    except Exception as e:
        logger.critical(f"Unexpected Runtime Error: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        run_search()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")