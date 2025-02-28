import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio as rio
import os

def normalize_image(img):
    """
    Normalize the image to have pixel values between -1 and 1 using Keras.
    
    Args:
        img: Input image as a NumPy array
    
    Returns:
        Normalized image as a NumPy array
    """
    img = img.astype(np.float32)

    # Maximum absolute height value in the dataset 
    # "Challenger Deep" in the Mariana Trench
    max_abs_data = 11000

    # Scale the image to be between -1 and 1
    # zero remains zero

    img = img / max_abs_data
    
    return img

def random_sample_patches(image_paths, square_size=256, num_patches=16):
    """
    Randomly sample patches from multiple images using TensorFlow.
    
    Args:
        image_paths: List of paths to the input images
        square_size: Size of each square patch (in pixels)
        num_patches: Number of patches to sample from each image
    
    Returns:
        List of lists of randomly sampled patches from each image
    """
    all_patches = []
    imgs = []
    for image_path in image_paths:
        # Open the image using rasterio
        with rio.open(image_path) as src:
            img_raw = src.read()  # Read all bands
            if src.count > 1:
                img_raw = np.stack([src.read(band) for band in range(1, src.count)], axis=0)
            else:
                img_raw = np.array(img_raw)
            print(image_path)
            print(img_raw.shape)

        img = np.array(img_raw)
        #img = normalize_image(img)
        imgs.append(img)

    # Generate a random starting point for cropping
    num_bands, img_height, img_width = imgs[0].shape
    xs_to_crop = np.random.randint(0, img_width - square_size, num_patches)
    ys_to_crop = np.random.randint(0, img_height - square_size, num_patches)
    print(xs_to_crop + square_size)

    for img in imgs:
        
        patches = []
        for i in range(num_patches):
            
            patch = img[:, ys_to_crop[i]:(ys_to_crop[i] + square_size), xs_to_crop[i]:(xs_to_crop[i] + square_size)]
            patches.append(patch)
        
        all_patches.append(patches)
    
    return all_patches

def visualize_patches(patches):
    """
    Visualize the split patches with a colorbar legend.
    
    Args:
        patches: List of patches to visualize contains a list of patches for each image
        num_cols: Number of columns in visualization grid
    """
    num_images = len(patches)
    num_patches = 6#len(patches[0])
    
    plt.figure(figsize=(15, 15))
    for img_idx in range(num_images):
        for patch_idx in range(num_patches):
            plt.subplot(num_images, num_patches, img_idx * num_patches + patch_idx + 1)
            if patches[img_idx][patch_idx].shape[0] == 3:  # Check if the patch has 3 bands (e.g., RGB)
                plt.imshow(np.transpose(patches[img_idx][patch_idx], (1, 2, 0)))  # Transpose to (height, width, channels)
            else:
                plt.imshow(patches[img_idx][patch_idx][0], cmap='gray')  # Display single band as grayscale
            plt.axis('off')
    plt.tight_layout()
    plt.show()


# Get all image paths in the folder
folder_path = "/home/tague/classes/aiclass/RawData/"
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tiff')]
square_size = 256

# Randomly sample patches
random_all_patches = random_sample_patches(image_paths, square_size, num_patches=16)

# Display randomly sampled patches
visualize_patches(random_all_patches)

# create a model for classification
# outputs an image with the same dimensions as the input image
# the image is a classification map
# each pixel is a class label
# the model is a convolutional neural network

# Create a simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
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
