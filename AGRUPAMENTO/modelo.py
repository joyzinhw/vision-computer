import torch
import torchvision.models as models
from torch import nn
import numpy as np
import os

def carregar_modelo_dispositivo(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(pretrained=True)
    modelo = nn.Sequential(*list(resnet.children())[:-1])
    modelo = modelo.to(device)
    modelo.eval()
    return modelo, device

def extrair_features_batch(modelo, imagens_tensor, device, batch_size=32, salvar_em=None):
    features = []
    modelo.eval()
    with torch.no_grad():
        num_batches = (len(imagens_tensor) + batch_size - 1) // batch_size
        for i in range(0, len(imagens_tensor), batch_size):
            batch = imagens_tensor[i:i+batch_size].to(device)
            feats = modelo(batch)
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            features.append(feats)
            print(f"Processando batch {i//batch_size + 1}/{num_batches}")
    features = np.vstack(features)
    
    if salvar_em:
        np.save(salvar_em, features)
        print(f"Features salvas em {salvar_em}")
        
    return features

def carregar_features(caminho):
    if os.path.exists(caminho):
        print(f"Carregando features de {caminho}")
        return np.load(caminho)
    return None
