"""
Segment tissue regions of low magnification whole slide images in the dataset.
"""

import os
import logging
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from slideloader import SlideLoader
from slidesegmenter import SlideSegmenter


input_folder = r""
image_output_folder = r""
segmentation_output_folder = r""
log_dir = r""

segmentation_magnification = 1.25
max_difference = 0.1
tissue_threshold = 'default'
pen_marking_threshold = 'default'


if __name__ == '__main__':

    # check if output folder exists
    for output_folder in [image_output_folder, segmentation_output_folder]:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    # configure logging
    log_dir = Path(log_dir)
    start = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    logging.basicConfig(
        level=logging.INFO,
        filename=log_dir/f'{start}_segmentation_log.txt',
        format='%(asctime)s - %(message)s',
        datefmt='%d/%m/%Y %I:%M:%S %p',
        encoding='utf-8',
    )
    logger = logging.getLogger(__name__)

    # find the unique filenames
    unique_filenames = {}
    filenames = os.listdir(input_folder)
    for filename in filenames:
        stripped_filename = filename
        for extension in ['.1.dcm', '.2.dcm', '.3.dcm', '.4.dcm', '.ndpi', '.svs']:
            stripped_filename = stripped_filename.replace(extension, '')
        
        if stripped_filename not in unique_filenames:
            unique_filenames[stripped_filename] = [filename]
        else:
            unique_filenames[stripped_filename].append(filename)

    # determine the device for segmentation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # change paths to Path instances
    input_folder = Path(input_folder)
    image_output_folder = Path(image_output_folder)
    segmentation_output_folder = Path(segmentation_output_folder)

    # create loader and segmenter instances
    loader = SlideLoader({'max_difference': 0.49/4096})
    segmenter = SlideSegmenter(            
        channels_last = True,
        tissue_segmentation = True,
        pen_marking_segmentation = True,
        separate_cross_sections = True,
        model_folder = 'latest',
        device = device
    )
    # loop over files and perform the segmentation
    for i, (unique_name, filenames) in enumerate(unique_filenames.items()):

        # construct the file paths
        file_paths = [input_folder/filename for filename in filenames]

        # load the slide and determine the exact magnification for reading 
        # the image at a low magnification
        try:
            loader.load_slide(file_paths)
        except: 
            logger.info(f'{i} - Loading {filenames} was unsuccessful.')
            continue

        # read the image
        try:
            image = loader.get_image(magnification=segmentation_magnification)
        except:
            logger.info(f'{i} - Getting image for {filenames} was unsuccessful.')
            loader.close()
            continue
        else:
            loader.close()

        logger.info(f'{i} - Finished loading {filenames} at {segmentation_magnification:0.2f}x')

        # segment the tissue (and pen markings) regions on the image
        try:
            segmentation = segmenter.segment(
                image=image/255, 
                tissue_threshold=tissue_threshold,
                pen_marking_threshold=pen_marking_threshold,
            )
        except torch.cuda.OutOfMemoryError:
            logger.info('Segmentation is performed on the CPU (not enough GPU memory available)')
            segmenter.change_device('cpu')
            segmentation = segmenter.segment(
                image=image/255,
                tissue_threshold=tissue_threshold,
                pen_marking_threshold=pen_marking_threshold,
            )
            segmenter.change_device(device)
        except Exception as error:
            logger.info(f'{i} - Segmentation of {filenames} was unsuccessful.')
        else:
            logger.info(f'{i} - Finished segmentation of {filenames}.')

        # save image
        image_output_path = image_output_folder/f'{unique_name}.png'
        sitk.WriteImage(sitk.GetImageFromArray(image[None, ...]), image_output_path)

        # process segmentation to make it suitable for saving
        segmentation = np.concatenate([segmentation['tissue'], segmentation['pen']], axis=-1)
        segmentation = (segmentation*255).astype(np.uint8)
        segmentation = segmentation.transpose((2,0,1))[..., None]

        # save segmentation
        segmentation_output_path = segmentation_output_folder/f'{unique_name}.tiff'
        sitk.WriteImage(sitk.GetImageFromArray(segmentation), segmentation_output_path)

        # Append an existing TIFF file to an existing or new zip
        with zipfile.ZipFile(output_folder/"segmentations.zip", "a", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(segmentation_output_path, arcname=f"{unique_name}.tiff")
        os.remove(segmentation_output_path)

        logger.info(f'{i} - Saved segmentation of {filenames}.')