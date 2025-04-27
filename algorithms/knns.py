import numpy as np
from collections import Counter

# Train data
X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 1],
    [6, 5],
    [7, 8],
    [8, 6]
])

y_train = np.array([0, 0, 0, 1, 1, 1])

# New data point to be classfied
x_new = np.array([5, 5])

# Compute Euclidean distances from x_new to all X_train points
distances = []
for x in X_train:
    distance = np.sqrt(np.sum((x - x_new) ** 2))
    distances.append(distance)

# Sort distances and get the indices of the k nearest neighbors (k=3 for this example)    
k = 3
k_indices = np.argsort(distances)[:k]
k_nearest_labels = y_train[k_indices]

# Get the most common class label among the k nearest neighbors
most_common = Counter(k_nearest_labels).most_common(1)
predicted_class = most_common[0][0]

print(f"Predicted class is: {predicted_class}")
