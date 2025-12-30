import sys
import os
from typing import List
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

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

def create_embeddings() -> None:
    """
    Generates embeddings for documents using a SentenceTransformer model.
    Includes data cleaning and reproducibility steps.
    """
    set_seed(42)

    try:
        validate_input_file(config.DOCUMENTS_PATH)
        validate_csv_columns(config.DOCUMENTS_PATH, required_columns=['ID', 'Text'])
        ensure_directory_exists(config.EMBEDDINGS_DIR)
    except (FileNotFoundError, PermissionError, ValueError, OSError) as e:
        logger.error(f"Validation Error: {e}")
        return
    
    logger.info("Starting embedding generation pipeline...")
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Device: {config.DEVICE}")

    try:
        logger.info(f"Loading texts from: {config.DOCUMENTS_PATH}")
        df: pd.DataFrame = pd.read_csv(config.DOCUMENTS_PATH)
        
        logger.info("Preprocessing texts (cleaning whitespace/formatting)...")
        df['Text'] = df['Text'].apply(clean_text)
        
        doc_ids: List[str] = df['ID'].tolist()
        texts: List[str] = df['Text'].tolist()
        logger.info(f"Loaded and cleaned {len(texts)} documents.")

        logger.info(f"Loading model: {config.MODEL_NAME}...")
        model: SentenceTransformer = SentenceTransformer(config.MODEL_NAME, device=config.DEVICE)

        logger.info("Encoding documents...")
        embeddings: np.ndarray = model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=config.BATCH_SIZE
        )

        logger.info("Saving embeddings...")
        
        with open(config.DOC_EMBEDDINGS_PATH, "wb") as f:
            pickle.dump({'ids': doc_ids, 'embeddings': embeddings}, f)

        logger.info(f"Saved embeddings to: {config.DOC_EMBEDDINGS_PATH}")
        logger.info(f"Shape: {embeddings.shape}")

    except Exception as e:
        logger.critical(f"Unexpected Runtime Error: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        create_embeddings()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user (Ctrl+C).")