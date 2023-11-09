import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import csv

# Load face images from the zip file
faces = {}
with zipfile.ZipFile("face-dataset.zip") as facezip:
    for filename in facezip.namelist():
        if not filename.endswith(".pgm"):
            continue
        with facezip.open(filename) as image:
            faces[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

# Retrieve dimensions of the images
faceshape = list(faces.values())[0].shape

# Prepare training data
facematrix = []
facelabel = []
for key, val in faces.items():
    if key.startswith("s40/") or "10.pgm" in key:  # Skip class 40 and all 10th images, as they will be used as queries
        continue
    facematrix.append(val.flatten())
    facelabel.append(key.split("/")[0])
facematrix = np.array(facematrix)

# Perform PCA
from sklearn.decomposition import PCA
n_components = 200
pca = PCA(n_components=n_components)
pca.fit(facematrix)
eigenfaces = pca.components_

# Calculate weights for the training set
weights = eigenfaces @ (facematrix - pca.mean_).T

# Prepare CSV file for writing results
output_csv_path = "facial_recognition_results.csv"
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Query Image", "Best Match", "Euclidean Distance"])

    # Test with one out-of-sample image from each class
    for i in range(1, 41):
        query_key = f"s{i}/10.pgm"
        query = faces[query_key].reshape(1, -1)
        query_weight = eigenfaces @ (query - pca.mean_).T
        euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
        best_match_index = np.argmin(euclidean_distance)
        best_match_distance = euclidean_distance[best_match_index]

        # Write the results to the CSV file
        writer.writerow([query_key, facelabel[best_match_index], best_match_distance])
