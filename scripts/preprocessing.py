import re
import pandas as pd

# Δεν χρειάζεται import για το str, είναι built-in

def clean_text(text) -> str:
    """
    Cleans text for Semantic Search.
    
    Actions:
    1. Handles NaN/None values (prevents crashes).
    2. Removes excessive whitespace.
    3. Keeps punctuation (crucial for BERT models).
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
