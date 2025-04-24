import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt



def random_sample_patches(image_paths, square_size=256, num_patches=2048):
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
        print(f"Processing image: {image_path}")
        # Open the image using rasterio
        with rio.open(image_path) as src:
            img_raw = src.read()  # Read all bands
            if src.count > 1:
                img_raw = np.transpose(np.stack([src.read(band) for band in range(1, src.count)], axis=0), (1, 2, 0))
                if (image_path.endswith('Visible.tiff') or image_path.endswith('IR.tiff')):
                    img_raw = img_raw / 255.0 # Normalize to [0, 1]
                if (image_path.endswith('TreeCover.tiff')):
                    img_raw = np.mean(img_raw, axis=-1, keepdims=True) / 255.0  # Convert to greyscale by averaging across channels
            else:
                img_raw = np.transpose(np.array(img_raw), (1, 2, 0))
                img_raw = img_raw / 11000.0
                if image_path.endswith('Height.tiff'):
                    # Calculate the two-point central slope in x and y directions
                    slope_x = np.gradient(img_raw[:, :, 0], axis=1)  # Gradient along x-axis
                    slope_y = np.gradient(img_raw[:, :, 0], axis=0)  # Gradient along y-axis
                    slope = np.sqrt(slope_x**2 + slope_y**2)  # Combine gradients to get slope magnitude
                    slope = np.expand_dims(slope, axis=-1)  # Add a new axis to make it 3D
                    imgs.append(slope)
            
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
    
    all_patches = np.array(all_patches)
    print(f"all_patches shape: {all_patches.shape}")
    return all_patches

def load_images(image_paths):
    """
    Load an image using rasterio and normalize it.
    
    Args:
        image_path: Path to the input image
    """

    imgs = np.array([])
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        # Open the image using rasterio
        with rio.open(image_path) as src:
            img_raw = src.read()  # Read all bands
            if src.count > 1:
                img_raw = np.transpose(np.stack([src.read(band) for band in range(1, src.count)], axis=0), (1, 2, 0))
                if (image_path.endswith('Visible.tiff') or image_path.endswith('IR.tiff')):
                    img_raw = img_raw / 255.0 # Normalize to [0, 1]
            else:                    
                img_raw = np.transpose(np.array(img_raw), (1, 2, 0))
                img_raw = img_raw / 11000.0
                if image_path.endswith('Height.tiff'):
                    # Calculate the two-point central slope in x and y directions
                    slope_x = np.gradient(img_raw[:, :, 0], axis=1)  # Gradient along x-axis
                    slope_y = np.gradient(img_raw[:, :, 0], axis=0)  # Gradient along y-axis
                    slope = np.sqrt(slope_x**2 + slope_y**2)  # Combine gradients to get slope magnitude
                    slope = np.expand_dims(slope, axis=-1)  # Add a new axis to make it 3D
                    print(f"Image shape: {img_raw.shape}, Slope shape: {slope.shape}")
                    imgs = np.concatenate((imgs, slope), axis=-1)
            
            if imgs.size == 0:
                imgs = img_raw
            else:
                imgs = np.concatenate((imgs, img_raw), axis=-1)
    #imgs = np.concatenate(imgs, axis=-1)  # Convert list to NumPy array and concatenate along the last axis
    #print(f"all_patches shape: {imgs.shape}")
    return imgs

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