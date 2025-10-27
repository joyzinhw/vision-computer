import os
import cv2
import glob
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from darknet_yolov3 import Darknet
from hockey_dataset import HockeyDataset

BASE_DIR = "/home/joyzinhw/Documentos/tudo/DETECÇÃO/HockeyAI_Dataset/SHL"
DATASET_DIR = os.path.join(BASE_DIR, "dataset_yolo")
CFG_PATH = os.path.join(BASE_DIR, "yolov3.cfg")

IMG_SIZE = 416
EPOCHS = 100
BATCH = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = HockeyDataset(
    os.path.join(DATASET_DIR, "train/images"),
    os.path.join(DATASET_DIR, "train/labels"),
    img_size=IMG_SIZE
)
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

model = Darknet(CFG_PATH, img_size=IMG_SIZE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for imgs, targets in train_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        loss = sum([out.mean() for out in outputs])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Época [{epoch+1}/{EPOCHS}] - Loss médio: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), os.path.join(BASE_DIR, "yolov3_hockey.pth"))

model.eval()
test_img = random.choice(glob.glob(os.path.join(DATASET_DIR, "test/images", "*.jpg")))
img = cv2.imread(test_img)
img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
tensor = torch.tensor(img_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
with torch.no_grad():
    detections, _ = model(tensor)
print("Inferência concluída.")
