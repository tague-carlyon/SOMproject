import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import dataGathandMan


# Get all image paths in the folder
folder_path = "/home/tcarlyon/classes/aiclass/RawData/"
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tiff')]
square_size = 256

#imgs = dataGathandMan.stack_images(image_paths)

# Randomly sample patches
images = dataGathandMan.get_images(image_paths)

# Display randomly sampled patches
#visualize_patches(random_all_patches)

# create a model for classification
# outputs an image with the same dimensions as the input image
# the image is a classification map
# each pixel is a class label
# the model is a convolutional neural network
""""
# Create a simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 12)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])



# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
model.fit(random_all_patches[0], epochs=10)

# print the loss and accuracy
loss, accuracy = model.evaluate(random_all_patches[0])
print(f'Loss: {loss}, Accuracy: {accuracy}')

# output the classification map example for the first image
classification_map = model.predict(random_all_patches[0])
classification_map = np.argmax(classification_map, axis=-1)

figure = plt.figure(figsize=(15, 15))
plt.imshow(classification_map)
plt.axis('off')
plt.show()
"""