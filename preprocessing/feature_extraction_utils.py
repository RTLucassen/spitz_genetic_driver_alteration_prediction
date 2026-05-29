"""
Utility classes and functions for feature extraction.
"""

from math import ceil
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from slideloader import SlideLoader
from skimage.transform import rotate


class TileDataset(torch.utils.data.Dataset):
    
    max_difference = 0.1

    def __init__(
        self, 
        image_path: Union[str, Path, list[str], list[Path]], 
        tile_information: dict[str, Any],
        magnification: float = 20,
        channels_first: bool = True,
        chunk_shape: tuple[int, int] = (256, 256),
    ) -> None:
        """
        Initializes dataset instance to load tiles from whole slide images.

        Args:
            image_path:  Path to folder where all images are stored.
            tile_information:  Dictionary with tile information.
            magnification:  Magnification at which the tile is loaded.
            channels_first:  Indicates whether the channels dimension should be 
                before the spatial dimensions (i.e., True), or after the spatial
                dimensions (i.e., False).
            chunk_shape:  Shape of chunks to load the tile in pixels as (height, width).
        """
        # define instance attributes related to data
        self.image_path = image_path
        self.data_dict = tile_information

        # determine the correction offset for the position with respect to the centroid
        centroid = self.data_dict['centroid']
        min_distance = None
        centroid_position = None
        for position, location, shape, _ in self.data_dict['tiles']:
            if not self.data_dict['center']:
                location = [location[0]+(shape[0]/2), location[1]+(shape[1]/2)]
            distance = ((centroid[0]-location[0])**2+(centroid[1]-location[1])**2)**0.2
            if min_distance is None or distance < min_distance:
                min_distance = distance
                centroid_position = position
        self.centroid_offset = centroid_position

        # initialize slide loader, load the slide, and define instance attributes
        # related to settings
        self.loader = SlideLoader({'max_difference': 0.49/4096})
        self.loader.load_slide(self.image_path)
        self.magnification = magnification
        self.channels_first = channels_first
        self.chunk_shape = chunk_shape

    def __len__(self) -> int:
        return len(self.data_dict['tiles'])

    def __getitem__(self, index) -> Optional[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]:
        """
        Load a tile and prepare it.

        Args:
            index:  Index for selecting tile from the dataset.

        Returns:
            tile_index:  Torch tensor indicating the tile index, which is the 
                combination of the specimen index, image index, cross-section index, 
                and tile index.
            position:  Torch tensor indicating the cross-section index, 
                horizontal position, and vertical position.
            tile:  Whole slide image tile.
        """
        # get the tile information at the index position
        position, location, shape, _ = self.data_dict['tiles'][index]

        # increase the shape if correction for the rotation is required
        if self.data_dict['rotation'] != 0.0:
            new_shape = [int(shape[0]*1.5), int(shape[1]*1.5)]
        else:
            new_shape = shape

        # change the values if the coordinate refers to 
        # the center position of the tile
        if self.data_dict['center']:
            location = [
                round(location[0]-(new_shape[0]/2)),
                round(location[1]-(new_shape[1]/2)),
            ]
        # load the slide and determine the exact magnification for extraction
        magnification_levels = self.loader.get_properties()['magnification_levels']
        if (max(magnification_levels)+self.max_difference < self.magnification):
            raise ValueError('Maximum magnification available is smaller than '
                             'the specified extraction magnification.')
        else:
            exact_magnification = None
            for level in magnification_levels:
                if abs(self.magnification-level) < self.max_difference:
                    exact_magnification = level
            if exact_magnification is None:
                exact_magnification = self.magnification
        
        # read the tile       
        try:
            if (new_shape[0] <= self.chunk_shape[0]) and (new_shape[1] <= self.chunk_shape[1]):
                tile = self.loader.get_tile(
                    magnification=exact_magnification, 
                    location=tuple(location), 
                    shape=tuple(new_shape),
                    channels_last=True,
                )
            else:
                tile = self.__get_tile_in_chunks(
                    magnification=exact_magnification, 
                    location=location, 
                    shape=new_shape,
                    channels_last=True,
                    chunk_shape=self.chunk_shape,
                )
        except Exception as error:
            return None, None

        # rotate and crop tile if 
        if self.data_dict['rotation'] != 0.0:
            tile = rotate(tile, -self.data_dict['rotation'])
            top = round((tile.shape[0]-shape[0])/2)
            left = round((tile.shape[1]-shape[1])/2)
            tile = tile[top:top+shape[0], left:left+shape[1], :]

        # change position of channels dimension
        if self.channels_first:
            tile = np.transpose(tile, (2,0,1))
        
        # convert tile to torch Tensor 
        tile = torch.from_numpy(tile)
        
        if self.data_dict['rotation'] == 0.0:
            tile = tile / 255

        # convert to torch tensor
        position = torch.tensor((
            position[0]-self.centroid_offset[0], 
            position[1]-self.centroid_offset[1]),
        dtype=int)

        return tile, position

    def __get_tile_in_chunks(
        self,
        magnification: float,
        location: tuple[int, int],
        shape: tuple[int, int] = (256, 256), 
        chunk_shape: tuple[int, int] = (256, 256),
        channels_last: bool = True,
    ) -> np.ndarray:
        """
        Get a tile in chunks from the whole slide image based on the specified
        magnification, location, and shape.
        
        Args:
            magnification:  Magnification at which the whole slide image is loaded.
            location:  Location of top left pixel as (x, y) at the specified magnification.
            shape:  Shape of the tile in pixels as (height, width).
            channels_last:  Specifies if the channels dimension of the output tensor 
                is last. If False, the channels dimension is the second dimension.
            chunk_shape:  Shape of chunks to load the tile in pixels as (height, width).
        
        Returns:
            tile:  Whole slide image tile [uint8] as (height, width, channel) 
                for channels last or (channel, height, width) for channels first.
        """
        # determine locations and shapes of the chunks
        absolute_locations = []
        relative_locations = []
        shapes = []
        height, width = shape
        for y in range(ceil(height/chunk_shape[0])):
            for x in range(ceil(width/chunk_shape[1])):
                relative_locations.append((x*chunk_shape[1], 
                                           y*chunk_shape[0]))
                absolute_locations.append((location[0]+x*chunk_shape[1], 
                                           location[1]+y*chunk_shape[0]))
                shapes.append((
                    min(height, (y+1)*chunk_shape[0])-(y*chunk_shape[0]),
                    min(width, (x+1)*chunk_shape[1])-(x*chunk_shape[1]),
                ))
        # get chunks from the image at the requested magnification
        chunks = self.loader.get_tiles(
            magnification=magnification, 
            locations=absolute_locations,
            shapes=shapes,
            channels_last=channels_last,
        )
        # stitch the chunks together to obtain full tile
        tile_shape = (height, width, 3) if channels_last else (3, height, width)
        tile = np.zeros(tile_shape, dtype=np.uint8)

        # add values of the chunks to empty tile
        for (x, y), (height, width), chunk in zip(relative_locations, shapes, chunks):
            if channels_last:
                tile[y:y+height, x:x+width, :] = chunk
            else:
                tile[:, y:y+height, x:x+width] = chunk

        return tile


# feature preparation function
def prepare_features_for_saving(features: dict) -> dict:
    """
    Prepare features by converting them to lists for more efficient saving.

    Args:
        features:  Dictionary with feature vectors and corresponding positions.

    Returns:
        features:  Dictionary with feature vectors as lists and corresponding 
            positions (optionally corrected).
    """
    for level in features:
        for tile_index in features[level]:
            feature = features[level][tile_index]['feature']
            position = features[level][tile_index]['position']
            if len(feature.shape) == 1:
                features[level][tile_index]['feature'] = [feature.tolist()]
                features[level][tile_index]['position'] = [position]
            else:
                feature_list = feature.reshape((-1, feature.shape[0])).tolist()
                # correct tile positions
                _, tile_height, tile_width = feature.shape
                position_list = []
                for y in range(tile_height):
                    for x in range(tile_width):
                        position_list.append((
                            position[0],
                            (position[1]*tile_width)+x,
                            (position[2]*tile_height)+y,
                        )) 
                # save lists with features as lists and corrected positions
                features[level][tile_index]['feature'] = feature_list
                features[level][tile_index]['position'] = position_list
    
    return features

def custom_collate_fn(batch):
    batch = [item for item in batch if item != (None, None)]
    if len(batch):
        return torch.utils.data.dataloader.default_collate(batch)
    else:
        return None, None