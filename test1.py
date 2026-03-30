import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# SAME PERSON (Aaron_Sorkin)
a = np.load("test_features/Mix_face_grp/Aaron_Sorkin/Aaron_Sorkin_2.npy")
b = np.load("test_features/Mix_face_grp/Aaron_Sorkin/Aaron_Sorkin_3.npy")

print("Similarity SAME person:", cosine_similarity([a], [b])[0][0])

# DIFFERENT PERSON (replace with another real folder name)
c = np.load("test_features/Mix_face_grp/George_W_Bush/George_W_Bush_0.npy")

print("Similarity DIFFERENT person:", cosine_similarity([a], [c])[0][0])