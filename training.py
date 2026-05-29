"""
Implementation of whole slide image training.
"""

import io
import json
import logging
import os
import platform
import random
# Set environment variable to allow expandable CUDA memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchinfo import summary
from tqdm import tqdm

from model.ViT import ViT, ModifiedViT
from training_utils import (
    FeatureDataset, 
    FocalLoss,
)


split_col_name = ''
label_mappings = [
    {'ALK': 0, 'ROS1': 1, 'NTRK': 2, 'HRAS': 3, 'BRAF': 3, 'MAP3K8': 3},
    {'ALK': 0, 'ROS1': 1, 'NTRK': 2, 'HRAS': 3, 'BRAF': 4, 'MAP3K8': 5},
]

training_run_index = 0
for extractor_value in ["CONCHv1.5", "UNI", "UNI2", "Virchow2"]:
    for label_mapping in label_mappings:
        for center_value in ["UMCU", "CLB"]:
            for val_fold in range(5):

                training_run_index += 1

                # define training settings
                settings = {
                    "prefix": f"{training_run_index}_",
                    "suffix": f"_{len(set(label_mapping.values()))}_classes_TEST",
                    "center": center_value,
                    "feature_extractor": extractor_value,
                    "level": "specimen",
                    "label_mapping": label_mapping,
                    "stain_norm": None,
                    "orientation_norm": False,
                    "config_file": "config.json",
                    "subsets_file": "split.xlsx",
                    "subsets_config": {
                        "training": [j for j in range(5) if j != val_fold],
                        "validation": [val_fold],
                        "test": ["test"]
                    },
                    "seed": 12345,
                    "model": "ViT",
                    "model_config": {
                        "tile_shape": None,
                        "model_arguments": {
                            "patch_shape": None,
                            "embed_dim": 192,
                            "depth": 2,
                            "n_heads": 4,
                            "mlp_ratio": 4,
                            "use_pos_embed": True,
                            "dropout_prob": 0.0,
                            "attn_dropout_prob": 0.5,
                            "drop_path_rate": 0.0,
                            "pytorch_attn_imp": False,
                        },
                        "compile_model": False,
                    },
                    "training_config": {
                        "learning_rate": 2.5e-5,
                        "iterations": 32000,
                        "iterations_per_decay": 6400,
                        "decay_factor": 0.5,
                        "iterations_per_checkpoint": 320,
                        "iterations_per_update": 32,
                        "batch_size": 1,
                        "optimizer": "AdamW",
                        "loss_function": "CrossEntropyLoss",
                        "focal_gamma": 0.0,
                        "feature_dropout_prob": 0.5,
                        "maximum_features": 25000,
                        "early_stopping_patience": 15,
                        "num_workers": 0,
                        "pin_memory": False,
                        "device": "cpu",
                    },
                    "progress_bar": False,
                    "save_all_checkpoints": False,
                }

                settings_files = [
                    settings,
                ]

                # define directories
                data_directory = Path(r"")
                experiments_directory = Path(r"")
                split_directory = Path(r"")


                if __name__ == '__main__':

                    for setting_file in settings_files:

                        # load settings
                        if isinstance(setting_file, str):
                            with open(setting_file, 'r') as f:
                                settings = json.loads(f.read())

                        # construct feature folder name
                        tile_size = 512 if settings["feature_extractor"] == "CONCHv1.5" else 224
                        orientation_indication = "center_" if settings["orientation_norm"] else ""
                        if '+' in settings["center"]:
                            centers = settings["center"].split('+')
                        else:
                            centers = [settings["center"]]
                        
                        feature_directories = []
                        for center in centers:
                            feature_folder = f'{center}_tessellation_{tile_size}_{orientation_indication}stain_norm_{settings["stain_norm"]}'
                            feature_directories.append(data_directory/center/settings["feature_extractor"]/feature_folder)

                        # add the input embedding size to the settings
                        input_embedding_dict = {"CONCHv1.5": 768, "UNI": 1024, "UNI2": 1536, "Virchow2": 2560, "H-optimus-0": 1536}
                        settings["model_config"]["model_arguments"]["input_dim"] = input_embedding_dict[settings["feature_extractor"]]

                        # add the number of classes to the settings
                        settings["model_config"]["model_arguments"]["n_classes"] = len(set(settings["label_mapping"].values()))

                        # construct experiment name       
                        settings["experiment_name"] = f'{settings["prefix"]}{settings["center"]}_{settings["feature_extractor"]}_{orientation_indication}stain_norm_{settings["stain_norm"]}_fold-{settings["subsets_config"]["validation"][0]}_{settings["level"]}{settings["suffix"]}' 

                        # initialize logger
                        start = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
                        logging.basicConfig(
                            level=logging.INFO,
                            filename=f'{start}_{settings["experiment_name"]}_training_log.txt',
                            format='%(asctime)s - %(message)s',
                            datefmt='%d/%m/%Y %I:%M:%S %p',
                            encoding='utf-8',
                        )
                        logger = logging.getLogger(__name__)

                        # create experiment_folder
                        experiment_directory = experiments_directory / settings['experiment_name']
                        if experiment_directory.exists():
                            raise FileExistsError(
                                f'Experiment folder already exists: {settings["experiment_name"]}'
                            )
                        else:
                            os.makedirs(experiment_directory)
                            logger.info(f'Created directory: {experiment_directory}')
                            # create subfolder for checkpoints
                            checkpoint_directory = experiment_directory / 'checkpoints'
                            os.makedirs(checkpoint_directory)
                            logger.info(f'Created directory: {checkpoint_directory}')
                            
                        # save and log training settings
                        with open(experiment_directory / settings['config_file'], 'w') as f:
                            json.dump(settings, f)
                        logger.info(f'Training config: {str(settings)}')

                        # configure the device
                        device = settings["training_config"]["device"]
                        if device == 'cuda':
                            if torch.cuda.is_available():
                                logger.info(f'CUDA available: {torch.cuda.is_available()}')
                            else:
                                device = 'cpu'
                                logger.info('CUDA unavailable')
                        logger.info(f'Device: {device}')

                        # seed randomness
                        seed = settings['seed']+settings['subsets_config']['validation'][0]
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        torch.backends.cudnn.benchmark = True
                        logger.info(f'Using seed: {seed}')

                        # read subset information
                        training_dataset = {}
                        validation_dataset = {}
                        df = pd.read_excel(split_directory/settings['subsets_file'])
                        for feature_directory in feature_directories:
                            feature_names = os.listdir(feature_directory)
                            for feature_name in feature_names:
                                slide_name = '_'.join(feature_name.split('_')[:-1])
                                section_index = feature_name.split('_')[-1]
                                slide_info = df[df['slide_name'] == slide_name]
                                for partition, dictionary in [('training', training_dataset), ('validation', validation_dataset)]:
                                    if slide_info[split_col_name].iloc[0] in settings['subsets_config'][partition]:
                                        if settings["level"] == "specimen":
                                            key = str(slide_info['specimen_index'].iloc[0])
                                        elif settings["level"] == "slide":
                                            key = f'{slide_info["specimen_index"].iloc[0]}_{slide_info["slide_index"].iloc[0]}'
                                        elif settings["level"] == "section":
                                            key = f'{slide_info["specimen_index"].iloc[0]}_{slide_info["slide_index"].iloc[0]}_{section_index}'
                                        else:
                                            raise NotImplementedError
                                        
                                        if key not in dictionary:
                                            dictionary[key] = {
                                                'filenames': [feature_name],
                                                'directory': feature_directory.as_posix(),
                                                'center': slide_info['center'].iloc[0],
                                                'specimen_id': slide_info['specimen_id'].iloc[0],
                                                'patient_index': slide_info['patient_index'].iloc[0],
                                                'specimen_index': slide_info['specimen_index'].iloc[0],
                                                'slide_index': [slide_info['slide_index'].iloc[0]],
                                                'label': slide_info['label'].iloc[0],
                                                'weight': 1,
                                            }
                                        else:
                                            dictionary[key]['filenames'].append(feature_name)
                                            dictionary[key]['slide_index'].append(slide_info['slide_index'].iloc[0])


                        # initialize training dataset and dataloader instances
                        # NOTE: because the sequences of feature vectors differ per specimen,
                        # a batch size of 1 with gradient accumulation is used
                        training_dataset = FeatureDataset(
                            instance_information=training_dataset,
                            label_mapping=settings['label_mapping'],
                            iterations=settings['training_config']['iterations'],
                            feature_dropout_prob=settings['training_config']['feature_dropout_prob'],
                            maximum_features=settings['training_config']['maximum_features'],
                        )
                        train_dataloader = DataLoader(
                            dataset=training_dataset,
                            sampler= RandomSampler(training_dataset, replacement=False),
                            batch_size=settings['training_config']['batch_size'],
                            num_workers=settings['training_config']['num_workers'],
                            pin_memory=settings['training_config']['pin_memory'],
                        )

                        # initialize validation dataset and dataloader instances
                        validation_dataset = FeatureDataset(
                            instance_information=validation_dataset,
                            label_mapping=settings['label_mapping'],
                            iterations=None,
                            feature_dropout_prob=0.0,
                            maximum_features=settings['training_config']['maximum_features'],
                        )
                        validation_dataloader = DataLoader(
                            dataset=validation_dataset,
                            sampler= SequentialSampler(validation_dataset),
                            batch_size=1,
                            num_workers=settings['training_config']['num_workers'],
                            pin_memory=settings['training_config']['pin_memory'],
                        )

                        # record number of training and validation cases to check correctness
                        logger.info(f'Training {settings["level"]}s: {len(training_dataset.feature_names)}')
                        logger.info(f'Validation {settings["level"]}s: {len(validation_dataset.feature_names)}')

                        # select model type
                        if settings['model'] == 'ViT':
                            network = ViT
                        elif settings['model'] == 'ModifiedViT':
                            network = ModifiedViT
                        else:
                            ValueError(f'Invalid model name: {settings["model"]}')
                        
                        # initialize model
                        model = network(**settings['model_config']['model_arguments'])
                        # transfer the model to the selected device
                        model = model.to(device)

                        # capture model summary in variable
                        f = io.StringIO()
                        with redirect_stdout(f):
                            summary(model=model, depth=4, col_names=['num_params'])
                        logger.info('\n'+f.getvalue())
                        logger.info(model)
                            
                        # compile the model
                        if (settings['model_config']['compile_model'] 
                            and platform.system() == 'Linux'):
                            model = torch.compile(model)

                        # initialize optimizer
                        learning_rate = settings['training_config']['learning_rate']
                        if settings['training_config']['optimizer'] == 'AdamW':
                            optimizer = Adam(model.parameters(), lr=learning_rate)        
                        elif settings['training_config']['optimizer'] == 'Adam':
                            optimizer = AdamW(model.parameters(), lr=learning_rate)
                        else:
                            raise NotImplementedError(
                                'Specified optimizer not implemented: '
                                f'{settings["training_config"]["optimizer"]}.'
                            )

                        # initialize loss function
                        if settings['training_config']['loss_function'] == 'CrossEntropyLoss':
                            loss_function = FocalLoss(gamma=0)
                        elif settings['training_config']['loss_function'] == 'FocalLoss':
                            loss_function = FocalLoss(gamma=settings['training_config']['focal_gamma'])
                        else:
                            raise NotImplementedError(
                                'Specified loss function not implemented: '
                                f'{settings["training_config"]["loss_function"]}.'
                            )

                        # define progress bar if specified
                        progress = lambda x: (tqdm(x, total=len(training_dataset)) 
                                                if settings["progress_bar"] else x)

                        # start training loop
                        specimen_indices = []
                        accumulated_loss = []
                        training_loss = []
                        training_index = []
                        validation_loss = []
                        validation_index = []
                        best_validation_loss = None
                        for i, batch in progress(enumerate(train_dataloader)):
                            index = i+1
                            # update learning rate
                            if index % settings['training_config']['iterations_per_decay'] == 0:
                                learning_rate *= settings['training_config']['decay_factor']
                                for g in optimizer.param_groups:
                                    g['lr'] = learning_rate
                                logger.info(f'Iteration {str(index).zfill(7)} - '
                                                    f'Learning rate set to {learning_rate:.1e}')

                            # ---------------- TRAINING -------------------

                            # get the specimen index
                            specimen_indices.append(int(batch[0]))
                            batch = batch[1:-1]

                            # bring the data to the correct device
                            batch = [item.to(device) for item in batch]
                            x, pos, y, weight = batch

                            # get model prediction
                            y_pred = model(x, pos)

                            # calculate loss and account for gradient accumulation
                            loss = loss_function(y_pred, y, weight)
                            loss /= settings['training_config']['iterations_per_update']
                            accumulated_loss.append(loss.item())
                            
                            # perform the backwards pass
                            loss.backward()

                            # for debugging purposes
                            if False: 
                                for item in batch:
                                    print(item.shape)

                            if index % settings['training_config']['iterations_per_update'] == 0:
                                # update the network parameters and reset the gradient
                                optimizer.step()
                                optimizer.zero_grad() # set the gradient to 0 again

                                # calculate the training loss for the batch
                                training_loss.append(sum(accumulated_loss))
                                training_index.append(index)
                                logger.info((
                                    f'Iteration {str(index).zfill(7)} - '
                                    f'Training loss: {training_loss[-1]:0.4f} - '
                                    f'Specimen indices: {specimen_indices}'
                                ))
                                specimen_indices = []
                                accumulated_loss = []

                            # --------------- VALIDATION ------------------
                            # periodically evaluate on the validation set
                            if index % settings['training_config']['iterations_per_checkpoint'] == 0:

                                # set the model in evaluation mode
                                model.eval()


                                loss_per_image = []
                                # deactivate autograd engine (backpropagation not required here)
                                with torch.no_grad():
                                    for batch in validation_dataloader:

                                        # get the specimen index
                                        specimen_indices.append(int(batch[0]))
                                        batch = batch[1:-1]

                                        if True:
                                            # bring the data to the correct device
                                            batch = [item.to(device) for item in batch]
                                            x, pos, y, weight = batch

                                            # for debugging purposes
                                            if False: 
                                                for item in batch:
                                                    print(item.shape)

                                            # get model prediction
                                            y_pred = model(x, pos)

                                            # calculate loss
                                            loss = loss_function(y_pred, y, weight)
                                            loss_per_image.append(loss.item())

                                validation_loss.append(sum(loss_per_image)/len(loss_per_image))
                                validation_index.append(index)
                                logger.info((
                                    f'Iteration {str(index).zfill(7)} - '
                                    f'Validation loss: {validation_loss[-1]:0.4f} - '
                                    f'Specimen indices: {specimen_indices}'
                                ))
                                specimen_indices = []

                                # determine if the last model checkpoint achieved the best validation loss
                                save_checkpoint = False
                                if best_validation_loss is None:
                                    best_validation_loss = validation_loss[-1]
                                    save_checkpoint = True
                                elif best_validation_loss > validation_loss[-1]:
                                    best_validation_loss = validation_loss[-1]
                                    save_checkpoint = True
                                elif 'save_all_checkpoints' in settings:
                                    if settings['save_all_checkpoints']:
                                        save_checkpoint = True

                                # save model checkpoint
                                if save_checkpoint:
                                    torch.save({
                                        'iteration': index,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'val_loss': validation_loss[-1],
                                        },
                                        checkpoint_directory / f'checkpoint_I{str(index).zfill(7)}.tar',
                                    )

                                # set the model to training mode
                                model.train()

                                # check early stopping
                                if min(validation_loss[-min(settings['training_config']['early_stopping_patience'], len(validation_loss)):]) > best_validation_loss:
                                    logger.info(f"Early stopping with a patience of {settings['training_config']['early_stopping_patience']} triggered.")
                                    break

                        # save training and validation loss as excel file
                        train_loss_df = pd.DataFrame.from_dict({'index': training_index,
                                                                'loss': training_loss})
                        validation_loss_df = pd.DataFrame.from_dict({'index': validation_index,
                                                                        'loss': validation_loss})
                        
                        # create a excel writer object
                        with pd.ExcelWriter(experiment_directory / 'loss.xlsx') as writer:
                            train_loss_df.to_excel(writer, sheet_name='Training', index=False)
                            validation_loss_df.to_excel(writer, sheet_name="Validation", index=False)

                        # plot training and validation loss
                        fig, ax = plt.subplots()

                        # plot losses
                        ax.plot(training_index, training_loss, zorder=1,
                                color='royalblue', alpha=0.40, label=f'Train')
                        ax.plot(validation_index, validation_loss, 
                                color='forestgreen', zorder=2, label=f'Validation')
                        ax.scatter(validation_index, validation_loss, zorder=2, marker='o', 
                                    facecolor='white', edgecolor='forestgreen', lw=1.5, s=15)

                        # change axis setup
                        ax.set_xlabel('Iteration')
                        ax.set_ylabel('Loss')
                        ax.set_ylim(bottom=0)
                        plt.xlim([-25, training_index[-1]+25])
                        plt.legend()
                        plt.savefig(os.path.join(experiment_directory, 'loss.png'), dpi=300)