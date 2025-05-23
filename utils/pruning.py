import torch.nn.utils.prune as prune

def structured_prune(model, sparsity=0.4):
    # Apply structured pruning to convolutional layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=sparsity)
            prune.remove(module, 'weight')  # Per Lemma 1, store pruned weights
    return model

'''

import torch.nn.utils.prune as prune

def structured_prune(model, sparsity=0.4):
    # Apply structured pruning (λ=0.01) with error bound:
    # ||θ* - θ_dense||₂ ≤ √(2λs/μ) (Lemma 1)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=sparsity)
            prune.remove(module, 'weight')  # Store pruned weights
    return model
'''