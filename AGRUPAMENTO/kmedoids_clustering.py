import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
import matplotlib.image as mpimg
import os

from imagens import carregar_imagens, transformar_para_tensor
from modelo import carregar_modelo_dispositivo, extrair_features_batch, carregar_features

# ==========================
# Configurações
# ==========================
dataset_path = "AGRUPAMENTO/kvasir-dataset-v2"
num_clusters = 8
batch_size = 32
features_path = "features_kvasir.npy"
tamanho_imagem = (128,128)  # Reduz o tamanho para acelerar (opcional)

# ==========================
# Carregar imagens
# ==========================
imagens, caminhos = carregar_imagens(dataset_path)
print(f"Total de imagens: {len(imagens)}")

# Reduz o tamanho das imagens para acelerar
imagens_tensor = transformar_para_tensor(imagens, tamanho=tamanho_imagem)
print("Transformação para tensor concluída:", imagens_tensor.shape)

# ==========================
# Carregar features (se existir) ou extrair
# ==========================
features = carregar_features(features_path)

if features is None:
    modelo, device = carregar_modelo_dispositivo()
    print("Usando dispositivo:", device)
    features = extrair_features_batch(
        modelo, imagens_tensor, device, batch_size=batch_size, salvar_em=features_path
    )

print("Features carregadas/extraídas:", features.shape)

# ==========================
# Reduzir dimensionalidade
# ==========================
pca = PCA(n_components=50)
features_pca = pca.fit_transform(features)
print("Features reduzidas com PCA:", features_pca.shape)

# ==========================
# K-Medoids
# ==========================
kmedoids = KMedoids(n_clusters=num_clusters, random_state=42)
labels = kmedoids.fit_predict(features_pca)
print("Clusters encontrados:", np.unique(labels))

# ==========================
# Visualizar clusters
# ==========================
features_2d = PCA(n_components=2).fit_transform(features_pca)
plt.figure(figsize=(8,6))
for i in range(num_clusters):
    plt.scatter(features_2d[labels==i,0], features_2d[labels==i,1], label=f'Cluster {i}', alpha=0.6)
plt.legend()
plt.title("Clusters de imagens K-Medoids")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()

# ==========================
# Mostrar exemplos de cada cluster
# ==========================
for cluster_id in range(num_clusters):
    print(f"\nCluster {cluster_id}:")
    idxs = np.where(labels==cluster_id)[0][:5]
    plt.figure(figsize=(15,3))
    for i, idx in enumerate(idxs):
        img = mpimg.imread(caminhos[idx])
        plt.subplot(1,5,i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
import matplotlib.image as mpimg
import os

from imagens import carregar_imagens, transformar_para_tensor
from modelo import carregar_modelo_dispositivo, extrair_features_batch, carregar_features

# ==========================
# Configurações
# ==========================
dataset_path = "AGRUPAMENTO/kvasir-dataset-v2"
num_clusters = 8
batch_size = 32
features_path = "features_kvasir.npy"
tamanho_imagem = (128,128)  # Reduz o tamanho para acelerar (opcional)

# ==========================
# Carregar imagens
# ==========================
imagens, caminhos = carregar_imagens(dataset_path)
print(f"Total de imagens: {len(imagens)}")

# Reduz o tamanho das imagens para acelerar
imagens_tensor = transformar_para_tensor(imagens, tamanho=tamanho_imagem)
print("Transformação para tensor concluída:", imagens_tensor.shape)

# ==========================
# Carregar features (se existir) ou extrair
# ==========================
features = carregar_features(features_path)

if features is None:
    modelo, device = carregar_modelo_dispositivo()
    print("Usando dispositivo:", device)
    features = extrair_features_batch(
        modelo, imagens_tensor, device, batch_size=batch_size, salvar_em=features_path
    )

print("Features carregadas/extraídas:", features.shape)

# ==========================
# Reduzir dimensionalidade
# ==========================
pca = PCA(n_components=50)
features_pca = pca.fit_transform(features)
print("Features reduzidas com PCA:", features_pca.shape)

# ==========================
# K-Medoids
# ==========================
kmedoids = KMedoids(n_clusters=num_clusters, random_state=42)
labels = kmedoids.fit_predict(features_pca)
print("Clusters encontrados:", np.unique(labels))

# ==========================
# Visualizar clusters
# ==========================
features_2d = PCA(n_components=2).fit_transform(features_pca)
plt.figure(figsize=(8,6))
for i in range(num_clusters):
    plt.scatter(features_2d[labels==i,0], features_2d[labels==i,1], label=f'Cluster {i}', alpha=0.6)
plt.legend()
plt.title("Clusters de imagens K-Medoids")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()

# ==========================
# Mostrar exemplos de cada cluster
# ==========================
for cluster_id in range(num_clusters):
    print(f"\nCluster {cluster_id}:")
    idxs = np.where(labels==cluster_id)[0][:5]
    plt.figure(figsize=(15,3))
    for i, idx in enumerate(idxs):
        img = mpimg.imread(caminhos[idx])
        plt.subplot(1,5,i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
