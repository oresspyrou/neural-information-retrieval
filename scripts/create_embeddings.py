import sys
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scripts.logger_setup import setup_logger
from scripts.validator import validate_input_file, validate_csv_columns, ensure_directory_exists

try:
    logger = setup_logger()
except RuntimeError as e:
    print(f"CRITICAL: Logger setup failed: {e}")
    sys.exit(1)

def create_embeddings():
    """
    Generates embeddings for documents using a SentenceTransformer model.

    This function validates input files and directories, loads documents from a CSV,
    encodes them into embeddings, and saves the results to a pickle file.

    Logs progress and errors throughout the process. If validation fails, the function
    returns early without processing.

    Raises:
        No exceptions are raised directly; errors are logged and handled internally.
    """

    try:
        validate_input_file(config.DOCUMENTS_PATH)
        validate_csv_columns(config.DOCUMENTS_PATH, required_columns=['ID', 'Text'])
        ensure_directory_exists(config.EMBEDDINGS_DIR)
    except (FileNotFoundError, PermissionError, ValueError, OSError) as e:
        logger.error(f"Validation Error: {e}")
        return
    
    logger.info("Creating embeddings...")
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Device: {config.DEVICE}")

    try:
        logger.info(f"Loading texts from: {config.DOCUMENTS_PATH}")
        df = pd.read_csv(config.DOCUMENTS_PATH)
        
        doc_ids = df['ID'].tolist()
        texts = df['Text'].tolist()
        logger.info(f"Loaded {len(texts)} documents.")

        logger.info(f"Loading model: {config.MODEL_NAME}...")
        model = SentenceTransformer(config.MODEL_NAME, device=config.DEVICE)

        logger.info("Encoding documents...")
        embeddings = model.encode(
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
        logger.critical(f"🔥 Unexpected Runtime Error: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        create_embeddings()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user (Ctrl+C).")