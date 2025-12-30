import random
import os
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, Numpy, and PyTorch.
    
    Args:
        seed (int): The seed value to set (default is 42).
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False