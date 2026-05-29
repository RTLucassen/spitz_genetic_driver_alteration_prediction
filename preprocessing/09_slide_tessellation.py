"""
Tessellate the tissue cross-sections without orientation normalization based on the segmentations.
"""

import json
import os
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tifffile as tiff
from tqdm import tqdm

from tessellation_utils import tessellate, visualize_tessellation, combine


def squeeze_singletons(arr, axes, dims=('T','Z')):
    axes = list(axes)
    for d in dims:
        if d in axes:
            i = axes.index(d)
            if arr.shape[i] == 1:
                arr = np.take(arr, 0, axis=i)
                axes.pop(i)
    return arr, ''.join(axes)


# define settings for paths
center = 'CLB'
tile_dimension = 512

# define paths and settings
zip_path = Path(r"")/center/"segmentations.zip"
image_folder = Path(r"")/center
output_folder = Path(r"")/center/f"{center}_tessellation_{tile_dimension}"
visualization_folder = Path(r"")/f"visualization_{tile_dimension}"/center

segmentation_magnification = 1.25
extraction_magnification = 20
tile_shape = (tile_dimension, tile_dimension)
stride = (tile_dimension, tile_dimension)
min_tissue_fraction = 0.05
max_pen_marking_fraction = 0.005
save_visualization = False


if __name__ == '__main__':

    # check if output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # calculate the ratio
    ratio = extraction_magnification / segmentation_magnification

    # Open the zip
    slide_information = {'name': [], 'centroid': [], 'tiles': []}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for filename in tqdm(zf.namelist()):
            # Skip directories
            if filename.endswith("/"):
                continue
            # Only process TIFFs
            lower = filename.lower()
            if not (lower.endswith(".tif") or lower.endswith(".tiff")):
                continue

            # define filename of corresponding image
            image_filename = filename.split('_annotation')[0]+'.png'

            # read segmentation
            with zf.open(filename) as fileobj:
                # Read with tifffile to correctly handle multi-channel, pyramids, OME axes, etc.
                with tiff.TiffFile(fileobj) as tf:
                    series = tf.series[0] # pick primary series
                    array = series.asarray() # loads full multi-dimensional array
                    axes = getattr(series, "axes", "") # e.g. 'TCZYX', 'YXS', etc.

            # Squeeze obvious singleton dims like T/Z=1 to simplify
            array, _ = squeeze_singletons(array, axes, dims=('T','Z'))

            array = np.transpose(array, axes=(1,2,0))

            # get the height and width of the image at the segmentation and 
            # feature extraction magnification
            height, width, channels = array.shape

            tissue = array[..., 1:]
            pen_marking = array[..., 0]

            # (6) tessellate the tissue segmentation
            uncorrected_tile_information = tessellate(
                segmentation=tissue,
                shape=tuple([int(s/ratio) for s in tile_shape]),
                stride=tuple([int(s/ratio) for s in stride]),
                min_tissue_fraction=min_tissue_fraction,
                pen_marking_map=pen_marking,
                max_pen_marking_fraction=max_pen_marking_fraction,
                exceed_image=False,
            )

            # load the image
            if save_visualization:
                image = sitk.GetArrayFromImage(sitk.ReadImage(image_folder/image_filename))

                visualize_tessellation(
                    images=[image, combine(tissue, pen_marking)], 
                    tile_information=uncorrected_tile_information,
                    output_path=visualization_folder/image_filename,
                    line_color='hsv',
                    line_width=1,
                    downscale_factor=2,
                    axis=0,
                )
        
            # (8) convert the tile coordinates to the extraction magnification
            tile_information = {}
            if len(uncorrected_tile_information):
                for i, (cross_section, (centroid, tiles)) in enumerate(uncorrected_tile_information.items()):
                    # correct the centroid coordinate
                    corrected_centroid = (int(centroid[0]*ratio), int(centroid[1]*ratio))
                    # correct the tile coordinates
                    corrected_tiles = []
                    for position, location, shape, pen_marking_fraction in tiles:
                        corrected_tiles.append((
                            position, 
                            (int(location[0]*ratio), int(location[1]*ratio)),
                            (int(shape[0]*ratio), int(shape[1]*ratio)),
                            pen_marking_fraction,
                        ))
                    # combine results into a dictionary
                    tile_information = {
                        'centroid': corrected_centroid,
                        'rotation': 0.0,
                        'tiles': corrected_tiles,
                    }
                    # save the tile information
                    tessellation_filename = image_filename.replace('.png', '')+f'_{str(i).zfill(2)}.json'
                    with open(output_folder/tessellation_filename, 'w') as f:
                        f.write(json.dumps(tile_information))