import os
import random
import shutil
from glob import glob
from ultralytics import YOLO
import cv2

BASE_DIR = "/home/joyzinhw/Documentos/tudo/DETECÇÃO/HockeyAI_Dataset/SHL"
IMAGES_DIR = os.path.join(BASE_DIR, "frames")
LABELS_DIR = os.path.join(BASE_DIR, "annotations")
DATASET_DIR = os.path.join(BASE_DIR, "dataset_yolo")

os.makedirs(DATASET_DIR, exist_ok=True)
for split in ["train", "test"]:
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(DATASET_DIR, split, sub), exist_ok=True)

images = sorted(glob(os.path.join(IMAGES_DIR, "*.jpg")))
random.shuffle(images)
split_idx = int(0.8 * len(images))
train_imgs = images[:split_idx]
test_imgs = images[split_idx:]

def copy_pairs(img_list, split):
    for img_path in img_list:
        base = os.path.basename(img_path)
        label_name = os.path.splitext(base)[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_name)
        dest_img = os.path.join(DATASET_DIR, split, "images", base)
        dest_label = os.path.join(DATASET_DIR, split, "labels", label_name)
        shutil.copy(img_path, dest_img)
        if os.path.exists(label_path):
            shutil.copy(label_path, dest_label)

copy_pairs(train_imgs, "train")
copy_pairs(test_imgs, "test")

classes = ["player", "goalkeeper", "referee", "ball", "stick", "goal", "crowd"]
data_yaml = os.path.join(BASE_DIR, "hockey_dataset.yaml")

with open(data_yaml, "w") as f:
    f.write(f"""
train: {os.path.join(DATASET_DIR, 'train/images')}
val: {os.path.join(DATASET_DIR, 'test/images')}
nc: {len(classes)}
names: {classes}
""")

model = YOLO("yolov8n.pt")

model.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer='Adam',
    lr0=0.001,
    patience=15,
    project=os.path.join(BASE_DIR, "results"),
    name="yolov8_hockey_pipeline",
    verbose=True
)

metrics = model.val(split="val")
print("\nResultados de Avaliação:")
print(metrics.results_dict)

TEST_IMAGE = random.choice(test_imgs)
results = model.predict(source=TEST_IMAGE, conf=0.4, save=True, save_txt=True)

img = cv2.imread(TEST_IMAGE)
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    label = int(box.cls[0])
    conf = box.conf[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, f"{classes[label]} {conf:.2f}", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

save_path = os.path.join(BASE_DIR, "results", "prediction_example.jpg")
cv2.imwrite(save_path, img)

print(f"\nDetecção concluída: {save_path}")
print(f"Treino: {len(train_imgs)} imagens | Teste: {len(test_imgs)} imagens")
