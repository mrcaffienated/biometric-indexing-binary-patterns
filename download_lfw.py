from sklearn.datasets import fetch_lfw_people
import os
import cv2

# Download dataset
lfw = fetch_lfw_people(min_faces_per_person=2, resize=0.5)

X = lfw.images
y = lfw.target
names = lfw.target_names

print("Total subjects:", len(names))
print("Total images:", len(X))

# Create folder structure
base_dir = "lfw_subset"
os.makedirs(base_dir, exist_ok=True)

# Save only first 10 subjects
selected_subjects = list(range(10))

for img, label in zip(X, y):
    if label in selected_subjects:
        person_name = names[label].replace(" ", "_")
        person_dir = os.path.join(base_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)

        count = len(os.listdir(person_dir))
        if count < 8:  # Only take 2 images per subject
            img_path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
            img_uint8 = (img * 255).astype("uint8")
            cv2.imwrite(img_path, img_uint8)

print("LFW subset created.")