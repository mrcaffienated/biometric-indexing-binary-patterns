import os
import numpy as np
from deepface import DeepFace

# -------- SETTINGS --------
image_folder = "lfw_subset"
output_folder = "test_features"
embedding_model = "Facenet"   # use FaceNet
np.random.seed(42)

# -------- Extract Deep Embeddings --------
embeddings = []
labels = []

for person in os.listdir(image_folder):
    person_path = os.path.join(image_folder, person)
    if os.path.isdir(person_path):
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            try:
                rep = DeepFace.represent(
                    img_path=img_path,
                    model_name=embedding_model,
                    enforce_detection=False
                )[0]["embedding"]

                embeddings.append(rep)
                labels.append(person)

            except Exception as e:
                print("Error processing:", img_path)

embeddings = np.array(embeddings)
print("Embeddings shape:", embeddings.shape)

# -------- Cancelable Transformation (Random Projection) --------
embedding_dim = embeddings.shape[1]
R = np.random.randn(embedding_dim, embedding_dim)
np.save("projection_matrix.npy", R)
protected_templates = np.dot(embeddings, R)

# -------- Binary Conversion --------
threshold = np.median(protected_templates, axis=0)
binary_templates = (protected_templates > threshold).astype(int)

# -------- Save in Required FBP Structure --------
modalities = ["Mix_face_grp", "Mix_iris_grp"]

# -------- Save Face Modality --------
for idx, person in enumerate(labels):
    person_folder = os.path.join(output_folder, "Mix_face_grp", person)
    os.makedirs(person_folder, exist_ok=True)
    np.save(os.path.join(person_folder, f"{person}_{idx}.npy"),
            binary_templates[idx])

# -------- Create Slightly Different Iris Modality --------
noise = np.random.normal(0, 0.01, embeddings.shape)
iris_embeddings = embeddings + noise

R2 = np.random.randn(embedding_dim, embedding_dim)
iris_protected = np.dot(iris_embeddings, R2)

threshold2 = np.median(iris_protected, axis=0)
iris_binary = (iris_protected > threshold2).astype(int)

for idx, person in enumerate(labels):
    person_folder = os.path.join(output_folder, "Mix_iris_grp", person)
    os.makedirs(person_folder, exist_ok=True)
    np.save(os.path.join(person_folder, f"{person}_{idx}.npy"),
            iris_binary[idx])

print("Deep feature generation completed.")