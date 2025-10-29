# treino.py (com gráficos de treino e validação)
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from modelo import CNNRegressor
from PIL import Image
import matplotlib.pyplot as plt

# ======================
# 1. Configurações
# ======================
dataset_dir = "/home/joyzinhw/Documentos/TUDO/REGRESSÃO/dataset"
images_dir = os.path.join(dataset_dir, "images")
jsons_dir = os.path.join(dataset_dir, "jsons")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 20
batch_size = 16
lr = 1e-4
seed = 42
label_scale = 1000.0  # normalização das contagens

# ======================
# 2. Transformações
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======================
# 3. Filtrar imagens válidas
# ======================
valid_images = []
for img_name in sorted(os.listdir(images_dir)):
    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        base = os.path.splitext(img_name)[0]
        json_path = os.path.join(jsons_dir, base + ".json")
        if os.path.exists(json_path):
            valid_images.append(img_name)

print(f"✅ {len(valid_images)} imagens válidas encontradas com JSON correspondente.")
if len(valid_images) == 0:
    raise RuntimeError("Nenhuma imagem com JSON correspondente foi encontrada.")

# ======================
# 4. Dataset filtrado com normalização de labels
# ======================
class FilteredDataset(Dataset):
    def __init__(self, images_dir, jsons_dir, img_list, transform=None, scale=1.0):
        self.images_dir = images_dir
        self.jsons_dir = jsons_dir
        self.img_list = img_list
        self.transform = transform
        self.scale = scale

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        base = os.path.splitext(img_name)[0]
        json_path = os.path.join(self.jsons_dir, base + ".json")

        # carrega imagem
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # carrega anotação
        with open(json_path, 'r') as f:
            data = json.load(f)

        if 'human_num' in data:
            label = float(data['human_num'])
        elif 'count' in data:
            label = float(data['count'])
        elif 'valor' in data:
            label = float(data['valor'])
        else:
            if 'points' in data and isinstance(data['points'], list):
                label = float(len(data['points']))
            else:
                label = 0.0

        # normaliza label
        label = label / self.scale
        return image, torch.tensor(label, dtype=torch.float32)

# ======================
# 5. Cria dataset e divisão 80/20
# ======================
full_dataset = FilteredDataset(images_dir, jsons_dir, valid_images, transform=transform, scale=label_scale)
torch.manual_seed(seed)
n = len(full_dataset)
train_size = int(0.8 * n)
val_size = n - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print(f"Tamanho total: {n} | Treino: {len(train_dataset)} | Validação: {len(val_dataset)}")

# ======================
# 6. Modelo, perda e otimizador
# ======================
model = CNNRegressor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ======================
# 7. Avaliação
# ======================
def avaliar_modelo(model, data_loader, device, scale=1.0):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            outputs = model(imgs)
            y_true.extend((labels * scale).cpu().numpy())
            y_pred.extend((outputs * scale).cpu().numpy())
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    if len(y_true) < 2:
        return np.nan, np.nan, np.nan
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ======================
# 8. Histórico para gráficos
# ======================
history = {
    "train_loss": [],
    "val_mae": [],
    "val_rmse": [],
    "val_r2": []
}

# ======================
# 9. Loop de treinamento
# ======================
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Treinando Época {epoch+1}/{num_epochs}", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / max(1, len(train_loader))
    mae, rmse, r2 = avaliar_modelo(model, val_loader, device, scale=label_scale)

    history["train_loss"].append(avg_train_loss)
    history["val_mae"].append(mae)
    history["val_rmse"].append(rmse)
    history["val_r2"].append(r2)

    print(f"\nÉpoca [{epoch+1}/{num_epochs}]")
    print(f"→ Loss de Treino: {avg_train_loss:.4f}")
    print(f"→ MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

# ======================
# 10. Plot dos gráficos
# ======================
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(history["train_loss"], label="Train Loss")
plt.title("Loss de Treino")
plt.xlabel("Época")
plt.ylabel("MSE")
plt.grid(True)

plt.subplot(1,3,2)
plt.plot(history["val_mae"], label="Val MAE", color='orange')
plt.title("MAE Validação")
plt.xlabel("Época")
plt.ylabel("MAE")
plt.grid(True)

plt.subplot(1,3,3)
plt.plot(history["val_rmse"], label="Val RMSE", color='green')
plt.title("RMSE Validação")
plt.xlabel("Época")
plt.ylabel("RMSE")
plt.grid(True)

plt.tight_layout()
plt.show()

# ======================
# 11. Teste com uma imagem individual
# ======================
if len(valid_images) > 0:
    test_img = os.path.join(images_dir, valid_images[0])
    image = Image.open(test_img).convert('RGB')
    image_t = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(image_t)
    print(f"Predição de contagem para {os.path.basename(test_img)}: {pred.item() * label_scale:.2f}")
