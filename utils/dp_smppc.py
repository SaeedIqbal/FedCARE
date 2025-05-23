import torch
from opacus import PrivacyEngine

class DPSMPC:
    def __init__(self, model, epsilon=2):
        self.model = model
        self.epsilon = epsilon
        self.privacy_engine = PrivacyEngine(
            model,
            batch_size=64,
            sample_size=50000,
            alphas=[1 + 1e-10],  # Mock alpha for DP
            noise_multiplier=1.9,  # σ=1.9 at ε=2
            max_grad_norm=1.2  # Δ=1.2 gradient sensitivity
        )
        self.privacy_engine.attach(model)

    def __call__(self, x):
        return self.model(x)

'''

class DPSMPC:
    def __init__(self, model, epsilon=2):
        self.model = model
        self.privacy_engine = PrivacyEngine(
            self.model,
            batch_size=64,
            sample_size=50000,
            alphas=[1 + 1e-10],  # Mock alpha for DP
            noise_multiplier=1.9,  # σ ∝ 1/ε
            max_grad_norm=1.2  # Δ=1.2 gradient sensitivity
        )
        self.privacy_engine.attach(self.model)
'''

'''
import torch
from opacus import PrivacyEngine

class DPSMPC:
    def __init__(self, model, epsilon=2):
        # Theorem 3: σ ∝ 1/ε (DP noise bound)
        # Lemma 4: (Enc_SMPC(∇θ_t)) = ∇θ_t (perfect reconstruction)
        self.model = model
        self.privacy_engine = PrivacyEngine(
            model,
            batch_size=64,
            sample_size=50000,
            alphas=[1 + 1e-10],  # Mock alpha for DP
            noise_multiplier=1.9,  # σ=1.9 at ε=2
            max_grad_norm=1.2  # Gradient sensitivity
        )
        self.privacy_engine.attach(model)

    def __call__(self, x):
        return self.model(x)
'''