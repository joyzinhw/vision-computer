import os
import cv2
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image

class HockeyDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640):
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.labels = label_dir
        self.img_size = img_size

        # Normalização padrão ImageNet (para modelos pré-treinados)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.imgs)

    def letterbox(self, image, target_size=640):
        """Redimensiona mantendo a proporção e adiciona padding (como YOLO faz)."""
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = target_size - new_w, target_size - new_h
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(128, 128, 128))
        return padded, scale, left, top

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label_path = os.path.join(self.labels, os.path.basename(img_path).replace(".jpg", ".txt"))

        # --- Carrega e pré-processa imagem ---
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

        # Conversão para RGB e equalização de histograma (realce de contraste)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.GaussianBlur(img_rgb, (3, 3), 0)
        yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        img_rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

        orig_h, orig_w = img_rgb.shape[:2]

        # --- Redimensiona mantendo proporção (letterbox) ---
        img_resized, scale, pad_x, pad_y = self.letterbox(img_rgb, self.img_size)

        # --- Lê e converte anotações YOLO -> [x_min, y_min, x_max, y_max] ---
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls, x, y, w, h = map(float, line.split())
                    labels.append(int(cls))

                    # YOLO: coords normalizadas -> pixels
                    x *= orig_w
                    y *= orig_h
                    w *= orig_w
                    h *= orig_h

                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2

                    # Escala e padding (ajuste pós-letterbox)
                    x1 = x1 * scale + pad_x
                    x2 = x2 * scale + pad_x
                    y1 = y1 * scale + pad_y
                    y2 = y2 * scale + pad_y

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(self.img_size - 1, x2), min(self.img_size - 1, y2)

                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # --- Converte imagem para tensor e normaliza ---
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = self.normalize(img_tensor)

        target = {"boxes": boxes, "labels": labels}
        return img_tensor, target
