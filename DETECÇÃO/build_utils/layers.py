import torch
import torch.nn as nn

class FeatureConcat(nn.Module):
    """Concatena features de múltiplas camadas"""
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x, outputs):
        # Coleta todos os tensores das camadas especificadas
        tensors = []
        for layer_idx in self.layers:
            if layer_idx < len(outputs) and outputs[layer_idx] is not None:
                tensors.append(outputs[layer_idx])
        # Adiciona o tensor atual (x) se não estiver vazio
        if x is not None:
            tensors.append(x)
        
        if not tensors:
            raise ValueError("Nenhum tensor válido para concatenação")
            
        return torch.cat(tensors, dim=1)

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
                if layer < len(outputs) and outputs[layer] is not None:
                    x += outputs[layer] * self.w[i + 1]
        else:
            for layer in self.layers:
                if layer < len(outputs) and outputs[layer] is not None:
                    x += outputs[layer]
        return x