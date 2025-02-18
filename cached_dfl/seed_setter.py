# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:32:01 2024

@author: 94880
"""
import numpy as np
import random


# If you're using PyTorch
import torch

SEED = 10086

def set_seed():
    np.random.seed(SEED)
    random.seed(SEED)
    
    
    # For PyTorch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For CUDA

    # Additional settings for PyTorch to ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Ensure that PyTorch's deterministic mode is set (if you need reproducibility)
    # This can impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return SEED