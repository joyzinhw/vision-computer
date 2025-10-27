import os
import cv2
import glob
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class HockeyDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=416):
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.labels = label_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size))
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label_path = os.path.join(self.labels, os.path.basename(img_path).replace(".jpg", ".txt"))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        targets = torch.zeros((50, 5))
        if os.path.exists(label_path):
            with open(label_path) as f:
                for i, line in enumerate(f.readlines()):
                    c, x, y, w, h = map(float, line.strip().split())
                    targets[i] = torch.tensor([c, x, y, w, h])
        return img, targets
