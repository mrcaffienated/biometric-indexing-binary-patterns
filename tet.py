import numpy as np
import os

base = "test_features"

modalities = ["Mix_face_grp", "Mix_iris_grp"]
persons = ["001","002","003","004","005"]

for mod in modalities:
    for person in persons:
        path = os.path.join(base, mod, person)
        os.makedirs(path, exist_ok=True)

        for i in range(2):
            vec = np.random.randint(0,2,256)
            np.save(os.path.join(path, f"{person}_{i}.npy"), vec)

print("Structured dummy dataset created")