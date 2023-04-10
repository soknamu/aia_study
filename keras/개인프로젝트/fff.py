import os
import sys
import numpy as np
import skimage.draw
from mrcnn.config import Config
from mrcnn import model as modellib, utils


class InferenceConfig(Config):
    # Set batch size to 1 to run inference on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + object

    # Set the name of the model and path to the weights file
    NAME = "object"
    WEIGHTS_PATH = "path/to/weights.h5"

    # Set the image resizing parameters
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024


class ObjectSegmentation:
    def __init__(self):
        # Create a Mask R-CNN model in inference mode
        self.config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir=os.getcwd())

        # Load the weights file into the model
        self.model.load_weights(self.config.WEIGHTS_PATH, by_name=True)

    def segment(self, image):
        # Convert the image to a NumPy array
        image_arr = np.array(image)

        # Run object detection and semantic segmentation on the image
        results = self.model.detect([image_arr], verbose=0)

        # Get the segmentation mask for the first object in the image
        mask = results[0]['masks'][:,:,0]

        # Create a new RGBA image with the segmentation mask as the alpha channel
        alpha = (mask * 255).astype(np.uint8)
        rgba = np.concatenate([image_arr, np.expand_dims(alpha, axis=2)], axis=2)

        # Return the segmented image as a PIL Image object
        return Image.fromarray(rgba)