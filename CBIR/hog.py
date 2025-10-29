import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from skimage.feature import hog
import matplotlib.pyplot as plt

# ---------------------------
# Configurações
# ---------------------------
base_dir = "/home/joyzinhw/Documentos/tudo/CBIR/meduloblastoma infantil"
magnification = "100x"  # 10x ou 100x
top_k = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Modelo ResNet50 (feature extractor)
# ---------------------------
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove a camada final FC
resnet = resnet.to(device)
resnet.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_resnet_features(image_path):
    """Extrai features profundas da ResNet50."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(img_tensor).squeeze().cpu().numpy()
    return feat / np.linalg.norm(feat)


# ---------------------------
# HOG + Histograma de cor + Momentos de Hu
# ---------------------------
def extract_advanced_features(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 1️⃣ HOG
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys')
    
    # 2️⃣ Histograma de cor (32 bins por canal)
    color_hist = []
    for i in range(3):
        hist = cv2.calcHist([img_resized], [i], None, [32], [0,256])
        hist = cv2.normalize(hist, hist).flatten()
        color_hist.extend(hist)
    
    # 3️⃣ Momentos de Hu (forma/textura global)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hu_moments = cv2.HuMoments(cv2.moments(thresh)).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-12)
    
    return np.concatenate([hog_feat, color_hist, hu_moments])


# ---------------------------
# Extração de features
# ---------------------------
image_paths = []
features_list = []

for cls in os.listdir(os.path.join(base_dir, magnification)):
    cls_folder = os.path.join(base_dir, magnification, cls)
    if not os.path.isdir(cls_folder):
        continue
    for img_name in os.listdir(cls_folder):
        if img_name.lower().endswith(('.jpg','.png','.jpeg')):
            img_path = os.path.join(cls_folder, img_name)
            image_paths.append(img_path)
            
            # Extrai e combina as features
            res_feat = extract_resnet_features(img_path)
            add_feat = extract_advanced_features(img_path)
            features_list.append(np.concatenate([res_feat, add_feat]))

X_features = np.array(features_list)

# ---------------------------
# Normalização + PCA
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

n_components = min(50, X_scaled.shape[0], X_scaled.shape[1])
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X_scaled)

# ---------------------------
# Clustering (para re-ranking)
# ---------------------------
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X_reduced)

# ---------------------------
# Função CBIR
# ---------------------------
def retrieve_similar_images(query_path, top_k=5):
    query_res = extract_resnet_features(query_path)
    query_add = extract_advanced_features(query_path)
    query_feat = np.concatenate([query_res, query_add])
    
    query_scaled = scaler.transform([query_feat])
    query_reduced = pca.transform(query_scaled)
    
    dists = cdist(query_reduced, X_reduced, metric='euclidean').flatten()
    
    if query_path in image_paths:
        idx_query = image_paths.index(query_path)
        dists[idx_query] = np.inf
    
    cluster_id = clusters[image_paths.index(query_path)]
    same_cluster_idx = np.where(clusters == cluster_id)[0]
    cluster_dists = [(i, dists[i]) for i in same_cluster_idx if dists[i] != np.inf]
    cluster_dists.sort(key=lambda x: x[1])
    
    top_results = cluster_dists[:top_k]
    top_indices = [i for i,_ in top_results]
    
    print(f"\nTop-{top_k} imagens mais próximas de {os.path.basename(query_path)}:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"{rank}: {os.path.basename(image_paths[idx])} | distância: {dists[idx]:.4f}")
    
    return [image_paths[i] for i in top_indices]


# ---------------------------
# Teste e visualização
# ---------------------------
query_image = image_paths[0]
top_images = retrieve_similar_images(query_image, top_k=top_k)

plt.figure(figsize=(15,5))
plt.subplot(1, top_k+1, 1)
plt.imshow(cv2.cvtColor(cv2.imread(query_image), cv2.COLOR_BGR2RGB))
plt.title("Query")
plt.axis('off')

for i, img_path in enumerate(top_images):
    plt.subplot(1, top_k+1, i+2)
    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    plt.title(f"Rank {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()
