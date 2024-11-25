import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import os
import cv2
import numpy as np

def resize_and_save_images(image_folder, target_size=(224, 224), output_folder="resized_images_folder"):
    """
    Resize the original images and save them to a new folder.
    
    Parameters:
    - image_folder (str): Path to the folder containing the original images.
    - target_size (tuple): The size to which the images will be resized.
    - output_folder (str): The folder where resized images will be saved.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Resize the image to the target size
        image_resized = cv2.resize(image, target_size)
        
        # Save the resized image to the new folder
        resized_image_path = os.path.join(output_folder, image_file)
        cv2.imwrite(resized_image_path, image_resized)

    print(f"Resized images saved to: {output_folder}")


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np

def augment_images_from_folder(image_folder, output_folder, num_augmentations=5):
    """
    Augments images from a folder using random transformations and saves the augmented images.
    
    Parameters:
    - image_folder (str): Path to the folder containing the resized images.
    - output_folder (str): Path to the folder where augmented images will be saved.
    - num_augmentations (int): Number of augmented images to generate per original image.
    """
    # Set up ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,          # Randomly rotate the image by up to 20 degrees
        width_shift_range=0.2,      # Randomly shift the image horizontally
        height_shift_range=0.2,     # Randomly shift the image vertically
        shear_range=0.2,            # Shear angle in counter-clockwise direction
        zoom_range=0.2,             # Randomly zoom into the image
        horizontal_flip=True,       # Randomly flip the image horizontally
        fill_mode='nearest'         # Fill the empty areas after transformation
    )
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the list of images from the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Loop through each image and apply augmentation
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        
        # Check if image is read correctly
        if image is None:
            print(f"Error reading image: {image_file}")
            continue
        
        # Reshape image to have an additional dimension (for the generator to work)
        image_reshaped = image.reshape((1,) + image.shape)  # Add a batch dimension (1, height, width, channels)
        
        # Apply augmentation and save the generated images
        augmented_image_count = 0
        for batch in datagen.flow(image_reshaped, batch_size=1, save_to_dir=output_folder, 
                                  save_prefix=f"{image_file.split('.')[0]}_aug_", save_format='jpeg'):
            augmented_image_count += 1
            # Stop after generating the required number of augmented images
            if augmented_image_count >= num_augmentations:
                break

    print(f"Augmented images saved to: {output_folder}")
