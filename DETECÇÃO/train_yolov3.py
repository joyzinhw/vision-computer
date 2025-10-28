# ============================================================
# Arquivo: train_yolo_hockey.py
# ============================================================
import os
import cv2
import glob
import random
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

from darknet_yolov3 import Darknet

# ============================================================
# CONFIGURAÇÕES
# ============================================================
BASE_DIR = "/home/joyzinhw/Documentos/TUDO/DETECÇÃO/HockeyAI_Dataset/SHL"
DATASET_DIR = os.path.join(BASE_DIR, "dataset_yolo")
CFG_PATH = os.path.join(BASE_DIR, "yolov3.cfg")

IMG_SIZE = 320     # reduzido para menor uso de GPU
EPOCHS = 30
BATCH = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# DATASET
# ============================================================
class HockeyDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=416, max_targets=50):
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.labels_dir = label_dir
        self.img_size = img_size
        self.max_targets = max_targets
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label_path = os.path.join(self.labels_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        targets = torch.zeros((self.max_targets, 5), dtype=torch.float32)
        if os.path.exists(label_path):
            with open(label_path) as f:
                for i, line in enumerate(f.readlines()):
                    if i >= self.max_targets:
                        break
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    c, x, y, w, h = map(float, parts)
                    targets[i] = torch.tensor([c, x, y, w, h], dtype=torch.float32)
        return img, targets


# ============================================================
# LOSS SIMPLIFICADA
# ============================================================
def yolo_loss(outputs, targets):
    total_loss = 0
    for output in outputs:
        loss = torch.mean((output ** 2))
        total_loss += loss
    return total_loss


# ============================================================
# PREPARAÇÃO DO MODELO E DATALOADER
# ============================================================
train_dataset = HockeyDataset(
    os.path.join(DATASET_DIR, "train/images"),
    os.path.join(DATASET_DIR, "train/labels"),
    img_size=IMG_SIZE
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

model = Darknet(CFG_PATH, img_size=IMG_SIZE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print(f"✅ Modelo carregado com {sum(p.numel() for p in model.parameters()):,} parâmetros")
print(f"Treinando em: {DEVICE.upper()}")

# ============================================================
# LOOP DE TREINAMENTO COM MIXED PRECISION
# ============================================================
scaler = GradScaler()
model.train()

for epoch in range(EPOCHS):
    start_time = time.time()
    total_loss = 0.0
    batch_count = 0

    progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS}", unit="batch")

    for batch_idx, (imgs, targets) in enumerate(progress_bar):
        try:
            imgs = imgs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)
                loss = yolo_loss(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss / batch_count)

            del outputs, loss
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ Erro no batch {batch_idx}: {e}")
            continue

    elapsed = time.time() - start_time
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    print(f"🧠 Época [{epoch+1}/{EPOCHS}] finalizada | Loss médio: {avg_loss:.4f} | ⏱️ Tempo: {elapsed:.1f}s")

    # Salvar checkpoint a cada 5 épocas
    if (epoch + 1) % 5 == 0:
        ckpt_path = os.path.join(BASE_DIR, f"checkpoint_epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, ckpt_path)
        print(f"💾 Checkpoint salvo: {ckpt_path}")

# ============================================================
# SALVAR MODELO FINAL
# ============================================================
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch
}, os.path.join(BASE_DIR, "yolov3_hockey.pth"))
print("\n✅ Modelo final salvo como 'yolov3_hockey.pth'!\n")

# ============================================================
# TESTE DE INFERÊNCIA
# ============================================================
model.eval()
test_imgs = glob.glob(os.path.join(DATASET_DIR, "test/images", "*.jpg"))
if test_imgs:
    test_img = random.choice(test_imgs)
    img = cv2.imread(test_img)
    if img is not None:
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        tensor = torch.tensor(img_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.0

        with torch.no_grad():
            detections, _ = model(tensor)
        print(f"\n📸 Inferência concluída na imagem: {os.path.basename(test_img)}")
        print(f"Detections shape: {detections.shape}")
else:
    print("❌ Nenhuma imagem de teste encontrada")
