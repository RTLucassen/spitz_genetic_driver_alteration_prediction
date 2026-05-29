"""
Evaluation of the model performance.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
# Set environment variable to allow expandable CUDA memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from rotation_utils import RotationDataset, seed_worker, convert_to_angle_deg
from model.ViT import ViT


image_paths = Path(r"")
anno_paths = Path(r"")
model_folder = Path(r"")
out_path = Path(r"")
model_names = []

# define parameters and paths
SEED = 12345
overrule_device = 'cpu'


if __name__ == '__main__':

    # loop over experiment folders
    for model_name in model_names:

        # Set the random seed for reproducibility
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # load training settings
        model_path = model_folder/model_name
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            config = json.loads(f.read())
        config["model"]["tile_dropout_prob"] = 0.0

        # load the images, masks, and corresponding rotations
        dataset = RotationDataset(
            image_folder=image_paths, 
            segmentation_folder=anno_paths,
            config = {
                'apply_augmentation': False,
                'label_type': config['label_type'],
                'gaussian_stdev': config['gaussian_stdev'],
                'loss_function': config['loss_function'],
                'n_classes': config['model']['n_classes'],
            },
        )
        dataloader = DataLoader(
            dataset=dataset,
            sampler= SequentialSampler(dataset),
            batch_size=1,
            worker_init_fn=seed_worker,
            num_workers=0,
            pin_memory=False,
        )

        # define device
        if overrule_device is not None:
            device = overrule_device
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize the model
        model = ViT(**config['model'])
        # Load the pretrained weights
        last_checkpoint = sorted([name for name in os.listdir(model_path) if (name.endswith('.tar') or name.endswith('.pth'))])[-1]
        model_checkpoint = torch.load(os.path.join(model_path, last_checkpoint), map_location=device)
        if 'model_state_dict' in model_checkpoint:
            state_dict = model_checkpoint['model_state_dict']
        else:
            state_dict = model_checkpoint
        model.load_state_dict(state_dict, strict=True)

        # bring model to device(s)
        model.to(device)
        model.eval()
        
        if out_path.exists():
            df = pd.read_excel(out_path)
            filenames = df['filenames'].tolist()
            angle_predictions = df['angle_predictions'].tolist()  
        else:
            filenames = []
            angle_predictions = []    

        # loop over the images in the test set
        with torch.no_grad():
            for i in tqdm(range(len(filenames), len(dataset))):
                # get the image name, the image, and the positional information
                filename, image, pos = dataset.__getitem__(i)
                
                # skip case if filename is already present
                if filename in filenames:
                    continue

                # move the image and positional information to the correct device
                image = image[None, ...].to(device)
                pos = pos[None, ...].to(device)

                # forward pass, compute loss and backpropagate
                pred = model(image, pos).to('cpu')
                if config['loss_function'] == 'BCE':
                    pred = torch.sigmoid(pred[0:int(pred.shape[1]/2)])
                elif config['loss_function'] == 'CCE':
                    pred = torch.softmax(pred, dim=1)
                angle_pred = convert_to_angle_deg(pred, config['label_type'])

                filenames.append(filename)
                angle_predictions.append(angle_pred)
    
                df = pd.DataFrame.from_dict({'filenames': filenames, 'angle_predictions': angle_predictions})
                df.to_excel(out_path, index=False)