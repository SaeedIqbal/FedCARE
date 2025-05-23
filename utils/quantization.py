import torch
import torch.nn as nn
class QuantizedModel(nn.Module):
    def __init__(self, model, bits=8):
        super(QuantizedModel, self).__init__()
        self.model = model
        self.bits = bits
        self.scale = 1 / (2 ** (bits - 1))  # Dynamic range for 8-bit

    def forward(self, x):
        # Simulate 8-bit quantization
        x = x.mul(self.scale).round().div(self.scale)
        return self.model(x)

'''
class QuantizedModel(nn.Module):
    def __init__(self, model, bits=8):
        super(QuantizedModel, self).__init__()
        self.model = model
        self.bits = bits
        self.scale = 1 / (2 ** (bits - 1))  # Dynamic range for 8-bit
    
    def forward(self, x):
        x = x.mul(self.scale).round().div(self.scale)  # Quantization
        return self.model(x)
'''