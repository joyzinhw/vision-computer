import torch
import torch.nn as nn

class FeatureConcat(nn.Module):
    """Concatena features de múltiplas camadas"""
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers] + [x], dim=1)


class WeightedFeatureFusion(nn.Module):
    """Fusão de features com pesos"""
    def __init__(self, layers, weight=False):
        super().__init__()
        self.layers = layers
        self.weight = weight
        if weight:
            self.w = nn.Parameter(torch.ones(len(layers) + 1) / 2, requires_grad=True)

    def forward(self, x, outputs):
        if self.weight:
            x = x * self.w[0]
            for i, layer in enumerate(self.layers):
                x += outputs[layer] * self.w[i + 1]
        else:
            for layer in self.layers:
                x += outputs[layer]
        return x
