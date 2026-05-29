"""
Utility functions for orientation normalization.
"""

import os
import math
import random
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
  

# define rotation dataset class 
class RotationDataset(Dataset):

    def __init__(
        self, 
        image_folder: Union[str, Path], 
        segmentation_folder: Union[str, Path], 
        config: dict[str, Any],
    ) -> None:
        """
        Initializes the Rotation Dataset instance with the specified parameters.

        Args:
            image_folder: The path to the folder containing images.
            segmentation_folder: The path to the folder containing segmentations.
            config: A configuration dictionary containing additional settings.
        """
        self.image_folder = Path(image_folder)
        self.segmentation_folder = Path(segmentation_folder)
        self.filenames = list(os.listdir(self.image_folder))
        self.config = config

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx, predefined_angle=0):
        # get the selected filename
        filename = self.filenames[idx % len(self.filenames)]
        
        # construct the paths the to the image and segmentation
        image_path = self.image_folder/filename
        segmentation_path = self.segmentation_folder/filename
        
        # load the image and segmentation
        image = Image.open(image_path).convert("RGB")
        segmentation = Image.open(segmentation_path).convert("L")

        # prepare image as model input
        image, segmentation = rotate_image_with_correct_padding(
            image=image, 
            segmentation=segmentation, 
            angle=predefined_angle,
        )
        # convert to torch tensor and normalize
        image = torch.from_numpy(np.transpose(np.array(image), (2,0,1)))/255

        # Get the position of each patch in the image
        pos = get_pos(image, segmentation)      

        return filename, image, pos


def get_centroid_of_segmentation(segmentation):
    """ 
    Get the centroid of the segmentation.

    Parameters:
    segmentation (PIL.Image): The segmentation to find the centroid of.

    Returns:
    centroid (tuple): The (x, y) coordinates of the centroid. 
    """
    # Convert to numpy array
    segmentation_array = np.array(segmentation)

    # Get coordinates of foreground pixels (non-zero)
    y_indices, x_indices = np.where(segmentation_array > 0)

    # Compute centroid
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("Segmentation is empty — no foreground pixels found.")

    # Compute centroid
    x_centroid = x_indices.mean()
    y_centroid = y_indices.mean()
    centroid = (float(x_centroid), float(y_centroid))
    return centroid


def crop_to_segmentation(image, segmentation):
    """ 
    Crop the image and segmentation to the bounding box of the segmentation.
    
    Parameters:
    image (PIL.Image): The image to be cropped.
    segmentation (PIL.Image): The segmentation to be used for cropping.
    
    Returns:
    cropped_image (PIL.Image): The cropped image.
    cropped_segmentation (PIL.Image): The cropped segmentation. 
    """
    # Calculate the bounding box of the segmentation
    np_segmentation = np.array(segmentation)
    y_indices, x_indices = np.where(np_segmentation > 0)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return image, segmentation  # Avoid cropping everything
    left, right = x_indices.min(), x_indices.max()
    top, bottom = y_indices.min(), y_indices.max()
    box = (left, top, right + 1, bottom + 1)

    # Crop the image and segmentation to the bounding box of the segmentation
    cropped_image = image.crop(box)
    cropped_segmentation = segmentation.crop(box)

    return cropped_image, cropped_segmentation


def pad_to_fit_patches_with_centroid(image, segmentation, centroid, patch_size=16, bg_color=(245, 245, 245, 255)):
    """ 
    Pad the image to patchifyable size, such the the centroid is in the middle of patch (0,0). 
    Since there is no center pixel in the patch (16, 16), the centroid is pixel (8,8) (right bottom of 4 center pixels).
    
    Parameters:
        image (PIL.Image): The image to be padded.
        segmentation (PIL.Image): The segmentation to be padded.
        centroid (tuple): The (x, y) coordinates of the centroid.
        patch_size (int): The size of the patches. Default is 16.
        bg_color (tuple): The background color to use for padding. Default is (245, 245, 245, 255).
    
    Returns:
        padded_image (PIL.Image): The padded image.
        padded_segmentation (PIL.Image): The padded segmentation. 
    """
    # get image size
    w, h = image.size

    # Get the centroid coordinates
    cx, cy = centroid

    # Convert to pixel value
    cx = math.floor(cx)
    cy = math.floor(cy)

    # Caclulate target image dimensions
    number_extra_pixels_left = cx % patch_size
    if number_extra_pixels_left < patch_size // 2:
        add_left = patch_size // 2 - number_extra_pixels_left
    else:
        add_left = patch_size + patch_size // 2 - number_extra_pixels_left

    number_extra_pixels_top = cy % patch_size
    if number_extra_pixels_top < patch_size // 2:
        add_top = patch_size // 2 - number_extra_pixels_top
    else:
        add_top = patch_size + patch_size // 2 - number_extra_pixels_top

    number_extra_pixels_right = (w - cx) % patch_size
    if number_extra_pixels_right < patch_size // 2:
        add_right = patch_size // 2 - number_extra_pixels_right
    else:
        add_right = patch_size + patch_size // 2 - number_extra_pixels_right

    number_extra_pixels_bottom = (h - cy) % patch_size
    if number_extra_pixels_bottom < patch_size // 2:
        add_bottom = patch_size // 2 - number_extra_pixels_bottom
    else:
        add_bottom = patch_size + patch_size // 2 - number_extra_pixels_bottom

    # Create new padded image and segmentation
    padded_image = ImageOps.expand(image, border=(add_left, add_top, add_right, add_bottom), fill=bg_color)
    padded_segmentation = ImageOps.expand(segmentation, border=(add_left, add_top, add_right, add_bottom), fill=0)  

    return padded_image, padded_segmentation


def rotate_image_with_correct_padding(image, segmentation, angle, bg_color=(245, 245, 245, 255)):
    """ 
    Rotate the image by the given angle and return the rotated image. The background 
    color is set to (245, 245, 245). The image is centered on a square canvas.

    Parameters:
    image (PIL.Image): The image to be rotated.
    segmentation (PIL.Image): The segmentation to be rotated.
    angle (float): The angle to rotate the image.
    bg_color (tuple): The background color to use for padding. Default is (245, 245, 245, 255).

    Returns:
    padded_image (PIL.Image): The rotated image with padding.
    padded_segmentation (PIL.Image): The rotated segmentation with padding. 
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    # Remove background if not already remved
    background = Image.new("RGBA", image.size, bg_color)
    image = Image.composite(image, background, segmentation)
    
    # rotate the image if the angle is not equal to zero
    if angle == 0:
        cropped_image = image
        cropped_segmentation = segmentation
    else:
        rotated_image = image.rotate(-angle, expand=True, resample=Image.BILINEAR, fillcolor=bg_color)
        rotated_segmentation = segmentation.rotate(-angle, expand=True, resample=Image.NEAREST, fillcolor=0)

        # Trim the image and segmentation to remove unnecessary padding
        cropped_image, cropped_segmentation = crop_to_segmentation(rotated_image, rotated_segmentation)

    # Get the centroid of the segmentation
    centroid = get_centroid_of_segmentation(cropped_segmentation)

    # Pad the image and segmentation to next multiple of 16 with the centroid in the center of a patch
    padded_image, padded_segmentation = pad_to_fit_patches_with_centroid(cropped_image, cropped_segmentation, centroid, bg_color=bg_color)

    # Convert to RGB (remove alpha) and save
    padded_image = padded_image.convert("RGB")

    return padded_image, padded_segmentation


def get_pos(image, segmentation, patch_size=16):
    """ 
    Returns the position of each patch in the image. The position is calculated based on the centroid of the segmentation.

    Parameters:
        img (torch.Tensor): Image for position matrix needs to be calculated.
        segmentation (PIL.Image): Segmentation of the image.
        patch_size (int): Size of the patches. Default is 16.
    
    Returns:
        pos (torch.Tensor): Position matrix of the patches. 
    """
    # Calculate centroid and convert to pixel value
    centroid = get_centroid_of_segmentation(segmentation)
    cx = math.floor(centroid[0])
    cy = math.floor(centroid[1])

    # Get the size of the image and number of patches
    _, H, W = image.shape 
    h_patches = H // patch_size
    w_patches = W // patch_size

    # Convert pixel centroid to patch coords
    centroid_patch_x = int(cx // patch_size)
    centroid_patch_y = int(cy // patch_size)

    x_range = torch.arange(w_patches) - centroid_patch_x
    y_range = -(torch.arange(h_patches) - centroid_patch_y)

    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    pos = torch.stack([y_grid, x_grid], dim=2).reshape(-1,2)  # Stack into (H, W, 2) and then flatten to (H*W, 2)

    return pos


def seed_worker(worker_id):
    """ Set the random seed for each worker to ensure reproducibility. The seed is set based on the global RANDOM_SEED and the worker_id.
    
    Parameters:
        worker_id (int): ID of the worker.
    
    Returns:
        None. Sets the random seed for the worker. """
    np.random.seed(worker_id)
    random.seed(worker_id)


def coord_to_angle_deg(v):
    """ 
    Convert a 2D coordinate to an angle in degrees.

    Parameters:
        v (torch.Tensor): A 1D or 2D tensor representing the coordinate.
    
    Returns:
        float: The angle in degrees, normalized to the range [0, 360).
    """
    if v.dim() == 1:
        v = v / (v.norm() + 1e-8)
        return float((torch.atan2(v[1], v[0]) * 180 / torch.pi % 360).item())
    else:
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        return float((torch.atan2(v[:, 1], v[:, 0]) * 180 / torch.pi % 360).item())


def class_prob_to_angle_deg(pred) -> float:
    """
    Convert class probabilities to an angle in degrees using circular mean.
    Args:
        pred: Tensor of class probabilities with shape (batch, n_classes).
    Returns:
        float: Predicted angle in degrees.
    """
    pred = pred[0, ...]
    n_classes = len(pred)
    classes = torch.linspace(0, 360, n_classes+1)[:-1]
    # calculate the probability weighted coordinate
    sum_pred = torch.sum(pred)
    mean_x = torch.sum(torch.cos(classes/180*torch.pi)*pred)/sum_pred
    mean_y = torch.sum(torch.sin(classes/180*torch.pi)*pred)/sum_pred

    return coord_to_angle_deg(torch.tensor([mean_x, mean_y]))


def convert_to_angle_deg(pred, label_type: str) -> float:
    """
    Converts predictions to angle in degrees based on the label type.
    """
    if label_type == 'radians':
        return float(pred * 180 / torch.pi % 360)
    elif label_type == 'coordinate':
        return coord_to_angle_deg(pred)
    elif label_type == 'binary_prob':
        return class_prob_to_angle_deg(pred)
    elif label_type == 'class_prob':
        return class_prob_to_angle_deg(pred)
    else:
        raise NotImplementedError