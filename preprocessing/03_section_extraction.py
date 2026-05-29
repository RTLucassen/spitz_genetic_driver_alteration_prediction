"""
Extract images of the separate tissue cross-sections based on the segmentation.
"""


import io
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tifffile
from skimage.morphology import opening, diamond
from tqdm import tqdm


def get_white_pixel_bounding_box(image: np.ndarray, padding: int=0):
    """
    Given a binary or RGB image (NumPy array), return the bounding box
    (min_row, min_col, max_row, max_col) that contains all white pixels,
    with optional padding. Padding is clipped to image boundaries.

    Args:
        image:  Input image as a NumPy array (2D or 3D).
        padding:  Number of pixels to pad the bounding box.

    Returns:
        tuple:  (min_row, min_col, max_row, max_col) of the padded bounding box.
            Returns None if no white pixels are found.
    """
    height, width = image.shape[:2]

    if image.ndim == 3:
        white_mask = np.all(image == 255, axis=-1)
    else:
        white_mask = image == 255

    white_coords = np.argwhere(white_mask)
    if white_coords.size == 0:
        return None

    min_row, min_col = white_coords.min(axis=0)
    max_row, max_col = white_coords.max(axis=0)

    min_row = int(max(min_row - padding, 0))
    min_col = int(max(min_col - padding, 0))
    max_row = int(min(max_row + padding, height - 1))
    max_col = int(min(max_col + padding, width - 1))

    return (min_row, min_col, max_row, max_col)


# define paths and parameters
image_path = Path(r"")
anno_path = Path(r"")

out_image_path = Path(r"")
out_anno_path = Path(r"")

start = 0


if __name__ == '__main__':

    # get the paths to all files in the folders
    image_paths = sorted(list(image_path.iterdir()))
    anno_paths = sorted(list(anno_path.iterdir()))

    # loop over the pairs of images and annotations
    for index, annotation_path in tqdm(enumerate(anno_paths)):
        if index >= start:
            if '.zip' in annotation_path.as_posix():
                with zipfile.ZipFile(annotation_path, 'r') as zf:
                    with zf.open(annotation_path.name.replace('.zip', '.tiff')) as file:
                        annotation = tifffile.imread(io.BytesIO(file.read()))
            else:
                annotation = sitk.GetArrayFromImage(sitk.ReadImage(annotation_path))                    
            
            # add extra dimension if necessary
            if len(annotation.shape) == 2:
                annotation = annotation[None, ...]
            annotation = annotation.transpose((1,2,0))

            # isolate the dimensions
            pen_marking = annotation[..., 0:1]
            sections = annotation[..., 1:]

            filename = annotation_path.name.split('_annotation')[0]
            corresponding_image_path = None
            for path in image_paths:
                if filename in path.as_posix():
                    corresponding_image_path = path
            
            if corresponding_image_path is None:
                continue

            image = sitk.GetArrayFromImage(sitk.ReadImage(corresponding_image_path))
            # loop over the sections and extract them from the image and segmentation
            for i in range(sections.shape[-1]):
                new_image_name = f'{filename}_{str(i).zfill(2)}'

                if not (out_anno_path/(new_image_name+'.png')).exists():
                    # get the section annotation, increase it area, and get the bounding box
                    section_annotation = np.transpose(np.concatenate([sections[..., i:i+1], pen_marking], axis=-1), (2,0,1))
                    section_annotation = (section_annotation*255).astype(np.uint8)
                    opened_section_annotation = opening(section_annotation[0, ...], diamond(5))*255

                    out = get_white_pixel_bounding_box(opened_section_annotation, 30)
                    if out is None:
                        continue 
                    else:
                        min_row, min_col, max_row, max_col = out

                    sitk.WriteImage(sitk.GetImageFromArray(image[None, min_row:max_row, min_col:max_col, :]), out_image_path/(new_image_name+'.png'))
                    sitk.WriteImage(sitk.GetImageFromArray(section_annotation[: , min_row:max_row, min_col:max_col, None]*255), out_anno_path/(new_image_name+'.tiff'))