import sys
import os
import pickle
import pandas as pd
import subprocess
from typing import List, Dict, Set, Tuple

# --- CONFIGURATION ---
TREC_EVAL_EXE = r"C:\Users\user\Desktop\trec_eval\trec_eval.exe"

# --- SETUP PATHS ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from scripts.logger_setup import setup_logger

logger = setup_logger()

def load_qrels(filepath: str) -> Dict[str, Set[str]]:
    """Loads Ground Truth (QRELs) handling headerless files."""
    try:
       
        df = pd.read_csv(filepath, sep=None, engine='python', header=None)
        

        if len(df.columns) == 4:
            
            df.columns = ['QueryID', 'Iter', 'DocID', 'Relevance']
        elif len(df.columns) == 3:
           
            df.columns = ['QueryID', 'DocID', 'Relevance']
        elif len(df.columns) == 2:
           
            df.columns = ['QueryID', 'DocID']
        else:
            df = pd.read_csv(filepath, sep=None, engine='python')
            df.columns = df.columns.str.strip()

        df['QueryID'] = df['QueryID'].astype(str)
        df['DocID'] = df['DocID'].astype(str)
        
        qrels = {}
        for _, row in df.iterrows():
            qid = row['QueryID']
            docid = row['DocID']
            
            if 'Relevance' in df.columns:
                try:
                    rel = int(row['Relevance'])
                    if rel <= 0:
                        continue
                except:
                    pass 

            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].add(docid)
            
        return qrels
    except Exception as e:
        logger.error(f"Error loading QRELs: {e}")
        logger.error(f"Check if file exists and has correct format: {filepath}")
        sys.exit(1)

def export_trec_format(results: Dict[str, List[Tuple[str, float]]], run_id: str = "MY_SEMANTIC_SYSTEM"):
    """Creates the results file for trec_eval tool."""
    output_path = os.path.join(config.RESULTS_DIR, "trec_results.txt")
    logger.info(f"Creating Results file for trec_eval: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for q_id, doc_list in results.items():
            for rank, (doc_id, score) in enumerate(doc_list):
                f.write(f"{q_id} Q0 {doc_id} {rank+1} {score:.6f} {run_id}\n")
    return output_path

def export_qrels_trec_format(qrels: Dict[str, Set[str]]):
    """Creates the QRELS file for trec_eval tool."""
    output_path = os.path.join(config.RESULTS_DIR, "qrels_trec.txt")
    logger.info(f"Creating QRELS file for trec_eval: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for q_id, doc_ids in qrels.items():
            for doc_id in doc_ids:
                f.write(f"{q_id} 0 {doc_id} 1\n")
    return output_path

def run_trec_eval_tool(qrels_file: str, results_file: str):
    """Runs the official trec_eval.exe and SAVES output."""
    if not os.path.exists(TREC_EVAL_EXE):
        logger.warning(f"TREC_EVAL exe not found at: {TREC_EVAL_EXE}")
        return

    logger.info("="*50)
    logger.info("RUNNING OFFICIAL TREC_EVAL TOOL")
    logger.info("="*50)

    command = [
        TREC_EVAL_EXE,
        "-m", "map",
        "-m", "P.5,10,15,20",
        qrels_file,
        results_file
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        if result.returncode == 0:
            print("\n" + result.stdout)
            
            output_txt_path = os.path.join(config.RESULTS_DIR, "evaluation_results.txt")
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write("--- TREC_EVAL OFFICIAL RESULTS ---\n")
                f.write(result.stdout)
                
            logger.info(f"RESULTS SAVED TO: {output_txt_path}")
        else:
            logger.error("Error running trec_eval:")
            print(result.stderr)

    except Exception as e:
        logger.error(f"Failed to run subprocess: {e}")

def run_evaluation():
    logger.info("Starting Evaluation Pipeline...")
    
    results_path = os.path.join(config.RESULTS_DIR, "search_results.pkl")
    if not os.path.exists(results_path):
        logger.error("Results file not found. Run search.py first.")
        return

    with open(results_path, "rb") as f:
        system_results = pickle.load(f)

    logger.info(f"Loading QRELs from: {config.QRELS_PATH}")
    qrels = load_qrels(config.QRELS_PATH)

    trec_res_file = export_trec_format(system_results)
    trec_qrels_file = export_qrels_trec_format(qrels)

    run_trec_eval_tool(trec_qrels_file, trec_res_file)

if __name__ == "__main__":
    run_evaluation()