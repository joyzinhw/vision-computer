# dataset.py
import os
import json
from torch.utils.data import Dataset
from PIL import Image

class CrowdDataset(Dataset):
    def __init__(self, img_dir, json_dir, file_list=None, transform=None):
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.transform = transform

        # Lista todas as imagens válidas (que têm JSON correspondente)
        if file_list is None:
            all_imgs = [f for f in os.listdir(img_dir)
                        if f.endswith('.jpg') or f.endswith('.png')]
        else:
            with open(file_list, 'r') as f:
                all_imgs = [line.strip() for line in f.readlines()]

        # Mantém apenas as que têm JSON correspondente
        self.img_files = []
        for img in all_imgs:
            json_path = os.path.join(json_dir, os.path.splitext(img)[0] + '.json')
            if os.path.exists(json_path):
                self.img_files.append(img)

        print(f"[INFO] {len(self.img_files)} imagens válidas carregadas (com JSON correspondente).")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        json_name = os.path.splitext(img_name)[0] + '.json'
        json_path = os.path.join(self.json_dir, json_name)

        # Carrega imagem
        image = Image.open(img_path).convert('RGB')

        # Carrega rótulo (campo 'count')
        with open(json_path, 'r') as f:
            data = json.load(f)
            label = data.get('count', 0)

        if self.transform:
            image = self.transform(image)

        return image, label
