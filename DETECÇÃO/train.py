import os
import torch
from torch.utils.data import DataLoader, random_split
from modelo import get_model
from hockey_dataset import HockeyDataset
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==================== CONFIGURAÇÕES ==================== #
BASE_DIR = "/home/joyzinhw/Documentos/TUDO/DETECÇÃO/HockeyAI_Dataset/SHL"
IMG_DIR = os.path.join(BASE_DIR, "frames")
LABEL_DIR = os.path.join(BASE_DIR, "annotations")
NUM_CLASSES = 7
NUM_EPOCHS = 10
BATCH_SIZE = 2
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== DATASET ==================== #
dataset = HockeyDataset(IMG_DIR, LABEL_DIR, img_size=640)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ==================== MODELO ==================== #
model = get_model(num_classes=NUM_CLASSES)
model.to(DEVICE)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=LR)

train_losses, val_losses = [], []

print(f"\n🚀 Iniciando treinamento do RetinaNet no dispositivo: {DEVICE}")
print(f"📂 Dataset dividido: {train_size} treino | {val_size} validação | {test_size} teste\n")

# ==================== LOOP DE TREINAMENTO ==================== #
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = total_cls_loss = total_box_loss = 0.0
    print(f"\n=================== ÉPOCA {epoch+1}/{NUM_EPOCHS} ===================")

    # ----------- TREINAMENTO -----------
    for i, (imgs, targets) in enumerate(train_loader):
        imgs = [img.to(DEVICE) for img in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        if all(len(t["boxes"]) == 0 for t in targets):
            continue

        loss_dict = model(imgs, targets)
        cls_loss = loss_dict["classification"]
        box_loss = loss_dict["bbox_regression"]
        total = cls_loss + box_loss

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        total_loss += total.item()
        total_cls_loss += cls_loss.item()
        total_box_loss += box_loss.item()

        print(f"[Batch {i+1}/{len(train_loader)}] Cls: {cls_loss.item():.4f} | Box: {box_loss.item():.4f} | Total: {total.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"\n📊 Treino | Época {epoch+1}: Média Cls={total_cls_loss/len(train_loader):.4f} | Média Box={total_box_loss/len(train_loader):.4f} | Média Total={avg_train_loss:.4f}")

    # ----------- VALIDAÇÃO -----------
    model.eval()
    val_loss_total = 0.0
    iou_scores = []
    iou_per_class = {i: [] for i in range(NUM_CLASSES)}
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            val_loss_total += sum(loss for loss in loss_dict.values()).item()

            preds = model(imgs)
            for pred, target in zip(preds, targets):
                if len(pred["boxes"]) == 0 or len(target["boxes"]) == 0:
                    continue

                ious = torchvision.ops.box_iou(pred["boxes"], target["boxes"])
                iou_scores.append(ious.max(dim=1)[0].mean().item())

                # IoU por classe
                for cls in range(NUM_CLASSES):
                    mask_pred = pred["labels"] == cls
                    mask_true = target["labels"] == cls
                    if mask_pred.any() and mask_true.any():
                        iou_cls = torchvision.ops.box_iou(
                            pred["boxes"][mask_pred], target["boxes"][mask_true]
                        )
                        iou_per_class[cls].append(iou_cls.max(dim=1)[0].mean().item())

                all_preds.extend(pred["labels"].cpu().numpy())
                all_labels.extend(target["labels"].cpu().numpy())

    avg_val_loss = val_loss_total / len(val_loader)
    val_losses.append(avg_val_loss)
    mean_iou = np.mean(iou_scores) if iou_scores else 0
    mean_iou_per_class = {k: np.mean(v) if v else 0 for k, v in iou_per_class.items()}

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0

    print(f"🧪 Validação | Loss={avg_val_loss:.4f} | IoU={mean_iou:.4f} | Acc={accuracy:.4f}")
    print(f"📊 IoU por classe: {mean_iou_per_class}")

# ==================== SALVAR MODELO ==================== #
torch.save(model.state_dict(), os.path.join(BASE_DIR, "retinanet_hockey.pth"))
print("\n✅ Modelo salvo com sucesso!\n")

# ==================== CURVA DE LOSS ==================== #
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Treino")
plt.plot(val_losses, label="Validação")
plt.title("Evolução do Loss (Treino vs Validação)")
plt.xlabel("Época")
plt.ylabel("Loss Médio")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(BASE_DIR, "loss_curve.png"))
plt.close()
print("📉 Curva de perda salva em loss_curve.png")

# ==================== MATRIZ DE CONFUSÃO ==================== #
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Class {i}" for i in range(NUM_CLASSES)])
disp.plot(cmap="Blues", values_format="d")
plt.title("Matriz de Confusão")
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"))
plt.close()
print("📊 Matriz de confusão salva em confusion_matrix.png")

# ==================== TESTE FINAL COM PREDIÇÕES ==================== #
model.eval()
with torch.no_grad():
    imgs, _ = next(iter(test_loader))
    imgs = [img.to(DEVICE) for img in imgs]
    preds = model(imgs)

img_np = imgs[0].permute(1, 2, 0).cpu().numpy()
boxes = preds[0]["boxes"].cpu().numpy()
for (x1, y1, x2, y2) in boxes:
    cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

cv2.imwrite(os.path.join(BASE_DIR, "retinanet_prediction.jpg"),
            cv2.cvtColor((img_np * 255).astype("uint8"), cv2.COLOR_RGB2BGR))
print("🖼 Exemplo de predição salvo em retinanet_prediction.jpg\n")
