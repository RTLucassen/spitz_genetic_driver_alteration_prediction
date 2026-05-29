"""
Correct section tessellation coordinates by converting them from the cropped images
to the whole slide images.
"""

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def rotate_point(
    x: float, 
    y: float, 
    angle_degrees: float, 
    origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    ox, oy = origin
    angle_radians = math.radians(angle_degrees)

    cos_a = math.cos(angle_radians)
    sin_a = math.sin(angle_radians)

    tx = x - ox
    ty = y - oy

    rx = tx * cos_a - ty * sin_a
    ry = tx * sin_a + ty * cos_a

    return rx + ox, ry + oy


# define settings for paths
center = 'CLB'
tile_dimension = 224
display_coords = False

# define paths
slide_tessellation_folder = Path(r"")/center/f'{center}_tessellation_{tile_dimension}'
section_tessellation_folder= Path(r"")/center/f'{center}_tessellation_{tile_dimension}'
rotation_filename = ""
rotation_path = Path(r"")/center/rotation_filename
image_folder = Path(r"")/center
out_slide_tessellation_folder =  Path(r"")/center/f'{center}_tessellation_{tile_dimension}_center'


if __name__ == '__main__':

    # create output folder is it does not exist yet
    if not out_slide_tessellation_folder.exists():
        os.makedirs(out_slide_tessellation_folder)

    # load the rotations
    df_rotation = pd.read_excel(rotation_path)
    rotation_dict = {row['filename']: 0.0 if isinstance(row['rotation'], str) else row['rotation']  for _, row in df_rotation.iterrows()}

    # get the filenames in the folder and load them
    corrected_section_tessellation = {}
    filenames = os.listdir(section_tessellation_folder)
    for filename in tqdm(filenames):
        
        # load the section tessellation information
        with open(section_tessellation_folder/filename, 'r') as f:
            section_tessellation_dict = eval(f.read())

        # load the slide tessellation information
        with open(slide_tessellation_folder/filename, 'r') as f:
            slide_tessellation_dict = eval(f.read())

        # get the centroid of the cross-section on the slide
        slide_centroid = slide_tessellation_dict['centroid']
        
        # get the relative coordinates with respect to the centroid of the 
        # cross-section in the orientation normalized crop
        relative_center_coords = []
        section_centroid = section_tessellation_dict['centroid']
        for tile in section_tessellation_dict['tiles']:
            relative_center_coords.append((
                tile[1][0]-section_centroid[0]+(tile[2][0]/2),
                tile[1][1]-section_centroid[1]+(tile[2][1]/2),
            ))

        # apply the rotation to the relative coordinates
        angle = rotation_dict[filename.replace('.json', '.png')]
        rotated_relative_center_coords = []
        for coord in relative_center_coords:
            rotated_relative_center_coords.append(rotate_point(coord[0], coord[1], -angle))
        
        # get the absolute coordinates by combinding the relative coordinates 
        # with the centroid of the section on the slide
        rotated_center_x = []
        rotated_center_y = []
        for coord in rotated_relative_center_coords:
            rotated_center_x.append((coord[0]+slide_centroid[0]))
            rotated_center_y.append((coord[1]+slide_centroid[1]))

        rotated_coordinates = []
        for x, y in zip(rotated_center_x, rotated_center_y):
            rotated_coordinates.append([x, y])

        corrected_tile_information = []
        for tile, coord in zip(section_tessellation_dict['tiles'], rotated_coordinates):
            corrected_tile_information.append([tile[0], coord, tile[2], tile[3]])

        # combine results into a dictionary
        tile_information = {
            'centroid': slide_centroid,
            'rotation': angle,
            'tiles': corrected_tile_information,
        }

        # save the tile information
        with open(out_slide_tessellation_folder/filename, 'w') as f:
            f.write(json.dumps(tile_information))

        # display image and coordinates
        if display_coords:
            xs = []
            ys = []
            for tile in slide_tessellation_dict['tiles']:
                xs.append(tile[1][0])
                ys.append(tile[1][1])

            minx = min(xs)/16
            maxx = max(xs)/16
            miny = min(ys)/16
            maxy = max(ys)/16
            x, y = slide_centroid[0]/16, slide_centroid[1]/16 

            image = sitk.GetArrayFromImage(sitk.ReadImage(image_folder/('_'.join(filename.split('_')[:-1])+'.png')))
            plt.scatter([minx, minx, maxx, maxx, x],[miny, maxy, miny, maxy, y])
            plt.scatter([i/16 for i in rotated_center_x], [i/16 for i in rotated_center_y], color='orange')
            plt.imshow(image)
            plt.show()