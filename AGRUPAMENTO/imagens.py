import os
from PIL import Image
import torch
import torchvision.transforms as transforms

def carregar_imagens(root_folder):
    imagens = []
    caminhos = []
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    path = os.path.join(folder_path, filename)
                    img = Image.open(path).convert("RGB")
                    imagens.append(img)
                    caminhos.append(path)
    return imagens, caminhos

def transformar_para_tensor(imagens, tamanho=(224,224)):
    transform = transforms.Compose([
        transforms.Resize(tamanho),
        transforms.ToTensor(),
    ])
    return torch.stack([transform(img) for img in imagens])
