from rasterio.stack import stack
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import os

def stack_images(image_paths):
    """
    Stack multiple images into a single NumPy array using Rasterio.
    
    Args:
        image_paths: List of paths to the input images
    
    Returns:
        Stacked images as a NumPy array
    """

    # Convert list of images to a numpy array and stack along the last axis
    imgs = np.concatenate(imgs, axis=-1)
    return imgs

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

 
    imgs = []
    for image_path in image_paths:
        # Open the image using rasterio
        with rio.open(image_path) as src:
            img_raw = src.read()  # Read all bands
            if src.count > 1:
                img_raw = np.transpose(np.stack([src.read(band) for band in range(1, src.count)], axis=0), (1, 2, 0))
            else:
                img_raw = np.transpose(np.array(img_raw), (1, 2, 0))
            
            imgs.append(img_raw)

    # Generate a random starting point for cropping
    img_height, img_width = imgs[0].shape[:2]

    xs_to_crop = np.random.randint(0, img_width - square_size, num_patches)
    ys_to_crop = np.random.randint(0, img_height - square_size, num_patches)
    
    all_patches = []
    
    for i in range(num_patches):
        patches = []
        for img in imgs:
            patch = img[ys_to_crop[i]:(ys_to_crop[i] + square_size), xs_to_crop[i]:(xs_to_crop[i] + square_size), :]
            patches.append(patch)

        all_patches.append(np.concatenate(patches, axis=-1))
        #np.concatenate((all_patches, np.expand_dims(np.concatenate(patches, axis=-1), axis=0)), axis=0) if all_patches.size else np.expand_dims(np.concatenate(patches, axis=-1), axis=0)
    

    print(f"all_patches shape: {all_patches[0].shape}")
    return all_patches

def visualize_patches(patches):
    """
    Visualize the split patches with a colorbar legend.
    
    Args:
        patches: List of patches to visualize contains a list of patches for each image
    """
    num_patches = len(patches)
    num_images = patches[0].shape[-1]  # Number of images is the last dimension of each patch
    
    plt.figure(figsize=(15, 15))
    for patch_idx in range(num_patches):
        for img_idx in range(num_images):
            plt.subplot(num_patches, num_images, patch_idx * num_images + img_idx + 1)
            patch = patches[patch_idx][..., img_idx]
            if patch.shape[0] == 3:  # Check if the patch has 3 bands (e.g., RGB)
                plt.imshow(np.transpose(patch, (1, 2, 0)))  # Transpose to (height, width, channels)
            else:
                plt.imshow(patch[0], cmap='gray')  # Display single band as grayscale
            plt.axis('off')
    plt.tight_layout()
    plt.show()