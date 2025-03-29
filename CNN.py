import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import dataGathandMan
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# %%
# Get all image paths in the folder
folder_path = "/home/tague/classes/aiclass/RawData/"
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tiff')]
square_size = 256

# %%
# Randomly sample patches
random_all_patches = dataGathandMan.random_sample_patches(image_paths, square_size)
# Ensure the data shape is correct
print(f"random_all_patches shape: {random_all_patches.shape}")
# %%

# Generate random labels for demonstration purposes
num_samples = random_all_patches.shape[0]
num_classes = 10  # Specify the number of classes
labels = np.random.randint(0, num_classes, num_samples)
# %%
# Reshape the data for clustering
reshaped_patches = random_all_patches.reshape(random_all_patches.shape[0], -1)

# Standardize the data
scaler = StandardScaler()
standardized_patches = scaler.fit_transform(reshaped_patches)

# Perform k-means clustering
num_clusters = 10  # Specify the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(standardized_patches)

# Print the cluster labels for debugging
print(f"Cluster labels: {cluster_labels}")
# %%

# Create a simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(256, 256, 12)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display the model summary
model.summary()
# %%
# Train the model
model.fit(random_all_patches, labels, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(random_all_patches, labels)
print(f'Loss: {loss}, Accuracy: {accuracy}')
# %%

# Output the classification map example for the first image
classification_map = model.predict(random_all_patches[0:1])
# Print the shape of classification_map for debugging
print(f"classification_map shape: {classification_map.shape}")

# Ensure classification_map has the correct shape for displaying
#if len(classification_map.shape) == 3:
#    classification_map = classification_map[0]

figure = plt.figure(figsize=(15, 15))
plt.imshow(classification_map)
plt.axis('off')
plt.show()
# %%