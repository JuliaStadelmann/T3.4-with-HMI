import torch
from torch import Tensor
from typing import List

def _permute_tensors(input_tensors: List[Tensor]) -> List[Tensor]:
    permutation = torch.randperm(len(input_tensors))
    
    return [input_tensors[i] for i in permutation]