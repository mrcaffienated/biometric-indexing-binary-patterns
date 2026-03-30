import os
import cv2
import numpy as np
from sklearn.decomposition import PCA

# -------- SETTINGS --------
image_folder = "lfw_subset"
output_folder = "test_features"
embedding_dim = 20

images = []
labels = []

# -------- Load Images --------
for person in os.listdir(image_folder):
    person_path = os.path.join(image_folder, person)
    if os.path.isdir(person_path):
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            images.append(img.flatten())
            labels.append(person)

images = np.array(images)

print("Total images loaded:", len(images))

# -------- PCA Embedding --------
pca = PCA(n_components=embedding_dim)
embeddings = pca.fit_transform(images)

print("Embeddings shape:", embeddings.shape)

# -------- Random Projection (Cancelable Template) --------
np.random.seed(42)
R = np.random.randn(embedding_dim, embedding_dim)
protected_templates = embeddings

# -------- Binary Conversion --------
threshold = np.median(protected_templates, axis=0)
binary_templates = (protected_templates > 0).astype(int)

# -------- Save in Required Structure --------
modalities = ["Mix_face_grp", "Mix_iris_grp"]

for modality in modalities:
    for idx, person in enumerate(labels):
        person_folder = os.path.join(output_folder, modality, person)
        os.makedirs(person_folder, exist_ok=True)
        np.save(os.path.join(person_folder, f"{person}_{idx}.npy"),
                binary_templates[idx])

print("Feature generation completed.")