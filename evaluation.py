"""
Implementation of model evaluation.
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
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from statistics import mean
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score, 
    ConfusionMatrixDisplay,
    roc_curve, 
    roc_auc_score, 
)
from torch.utils.data import DataLoader, SequentialSampler
from torchinfo import summary
from tqdm import tqdm

from model.ViT import ViT
from training_utils import FeatureDataset


def bootstrapper(
    y_true: list[int], 
    y_pred: list[float], 
    eval_func: Callable,
    iterations: int,
    stratified: bool,
    confidence_level: float,
    seed: int,
    show_progress: bool = True,
    return_sample_results: bool = False,
) -> dict[str, list]:
    """
    Function to obtain bootstrap estimates of statistics based on labels and predictions.

    Args:
        y_true:  List with labels.
        y_pred:  List with predictions.
        eval_func:  Function that calculates and returns a score based on 
            a list of labels and predictions.
        iterations:  Number of bootstrap repetitions.
        stratified:  Indicates whether cases are sampled with stratification.
        confidence_level:  Level of confidence interval between 0 and 1.
        seed:  Random seed.
        show_progress:  Indicates whether a progress bar should be printed. 
        return_sample_results:  Indicates whether the results of all individual 
            bootstrap repetitions should be returned.

    Returns:
        bootstrap_results:  Mean and confidence interval of bootstrapped statistic.
        sample_results:  Optionally returned results of individual bootstrap repetitions.
    """
    # set seed
    random.seed(seed)

    # group per class
    classes = {}
    for i, label in enumerate(y_true):
        label = str(label)
        if label not in classes:
            classes[label] = [i]
        else:
            classes[label].append(i)

    # initialize iterator
    if show_progress:
        iterator = tqdm(range(iterations))
    else:
        iterator = range(iterations)

    # sample with replacement with or without stratification          
    sample_results = []
    for _ in iterator:
        # generate indices
        if stratified:
            sample_indices = []
            for values in classes.values():
                sample_indices.extend(
                    [values[random.randint(0, len(values)-1)] for _ in range(len(values))]
                )
        else:
            sample_indices = [random.randint(0, len(y_true)-1) for _ in range(len(y_true))]
        # select cases with replacement based on generated indices
        y_true_sampled = [y_true[i] for i in sample_indices]
        y_pred_sampled = [y_pred[i] for i in sample_indices]
        sample_results.append(eval_func(y_true_sampled, y_pred_sampled))

    if None in sample_results:
        sample_results.remove(None)
    
    # calculate mean and confidence intervals for bootstrap samples
    bootstrap_results = {
        'mean': float(np.mean(sample_results)),
        f'{confidence_level*100}% CI lower': float(np.quantile(sample_results, (1-confidence_level)/2)),
        f'{confidence_level*100}% CI upper': float(np.quantile(sample_results, 1-((1-confidence_level)/2))),
    }
    if return_sample_results:
        return bootstrap_results, sample_results
    else:
        return bootstrap_results


configs = [
    # for models trained on both UMCU and CLB data
    {
        "center": "UMCU",
        "subsets": ["validation"],
        "bootstrap": False,
        "stain_norm": "same",
    },
    {
        "center": "UMCU",
        "subsets": ["test"],
        "bootstrap": False,
        "stain_norm": "same",
    },
    {
        "center": "CLB",
        "subsets": ["validation"],
        "bootstrap": False,
        "stain_norm": "same",
    },
    {
        "center": "CLB",
        "subsets": ["test"],
        "bootstrap": False,
        "stain_norm": "same",
    },
    # for models trained on data from a single center
   {
       "center": "same",
       "subsets": ["validation"],
       "bootstrap": False,
       "stain_norm": "same",
   },
   {
       "center": "same",
       "subsets": ["test"],
       "bootstrap": False,
       "stain_norm": "same",
   },
   {
       "center": "other",
       "subsets": ["training", "validation", "test"],
       "bootstrap": False,
       "stain_norm": "same",
   },
   {
       "center": "other",
       "subsets": ["test"],
       "bootstrap": False,
       "stain_norm": "same",
   },
    # for evaluation of stain normalization
   {
       "center": "other",
       "subsets": ["training", "validation", "test"],
       "bootstrap": False,
       "stain_norm": 0,
   },
   {
       "center": "other",
       "subsets": ["training", "validation", "test"],
       "bootstrap": False,
       "stain_norm": 1,
   },
   {
       "center": "other",
       "subsets": ["training", "validation", "test"],
       "bootstrap": False,
       "stain_norm": 2,
   },
   {
       "center": "other",
       "subsets": ["training", "validation", "test"],
       "bootstrap": False,
       "stain_norm": 3,
   },
   {
       "center": "other",
       "subsets": ["training", "validation", "test"],
       "bootstrap": False,
       "stain_norm": 4,
   },
]

# define directories
data_directory = Path(r"")
experiments_directory = Path(r"")
split_directory = Path(r"")

split_col_name = ''
experiment_folders = [folder for folder in os.listdir(experiments_directory)]

confidence_level = 0.95
iterations = 10000
stratified = True
verbose = True
overwrite = False
save_sample_results = True
save_attention = False
suffix = ""
device = "cuda"


if __name__ == '__main__':

    # initialize logger
    start = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    logging.basicConfig(
        level=logging.INFO,
        filename=f'{start}_evaluation_log.txt',
        format='%(asctime)s - %(message)s',
        datefmt='%d/%m/%Y %I:%M:%S %p',
        encoding='utf-8',
    )
    logger = logging.getLogger(__name__)

    for experiment_folder in experiment_folders:
        print(experiment_folder) 
        for original_config in configs:
            config = original_config.copy()
            if config['center'] == "same":
                if experiment_folder.split('_')[1] == "UMCU":
                    config['center'] = "UMCU"
                else:
                    config['center'] = "CLB"

            elif config['center'] == "other":
                if experiment_folder.split('_')[1] == "UMCU":
                    config['center'] = "CLB"
                else:
                    config['center'] = "UMCU"

            # define and create results directory if necessary
            results_directory = experiments_directory/experiment_folder/f'results_{config['center']}_{config['stain_norm']}_{'_'.join(config['subsets'])}{suffix}'
            if not results_directory.exists():
                os.makedirs(results_directory)
            elif not overwrite:
                continue

            if save_attention:
                attn_directory = results_directory / 'attention_results'
                if not attn_directory.exists():
                    os.makedirs(attn_directory)
                elif not overwrite:
                    continue

            logger.info(f"Start evaluation of {experiment_folder} on {config['subsets']} from {config['center']} with stain norm {config['stain_norm']}")

            # load settings
            with open(experiments_directory/experiment_folder/'config.json', 'r') as f:
                settings = json.loads(f.read())

            # configure stain norm config
            if config['stain_norm'] == "same":
                config['stain_norm'] = settings['stain_norm']

            # construct feature folder name
            tile_size = 512 if settings["feature_extractor"] == "CONCHv1.5" else 224
            orientation_indication = "center_" if settings["orientation_norm"] else ""
            feature_folder = f'{config["center"]}_tessellation_{tile_size}_{orientation_indication}stain_norm_{config["stain_norm"]}'
            feature_directory = data_directory/config["center"]/settings["feature_extractor"]/feature_folder 

            # configure the device
            if device is None:
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

            # get the included subsets
            included_subsets = []
            for subset in config["subsets"]:
                included_subsets.extend(settings['subsets_config'][subset])

            # read subset information
            test_dataset = {}
            df = pd.read_excel(split_directory/settings['subsets_file'])
            feature_names = os.listdir(feature_directory)
            for feature_name in feature_names:
                slide_name = '_'.join(feature_name.split('_')[:-1])
                section_index = feature_name.split('_')[-1]
                slide_info = df[df['slide_name'] == slide_name]
                if slide_info[split_col_name].iloc[0] in included_subsets:
                    if settings["level"] == "specimen":
                        key = str(slide_info['specimen_index'].iloc[0])
                    elif settings["level"] == "slide":
                        key = f'{slide_info["specimen_index"].iloc[0]}_{slide_info["slide_index"].iloc[0]}'
                    elif settings["level"] == "section":
                        key = f'{slide_info["specimen_index"].iloc[0]}_{slide_info["slide_index"].iloc[0]}_{section_index}'
                    else:
                        raise NotImplementedError
        
                    if key not in test_dataset:
                        test_dataset[key] = {
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
                        test_dataset[key]['filenames'].append(feature_name)
                        test_dataset[key]['slide_index'].append(slide_info['slide_index'].iloc[0])

            # initialize validation dataset and dataloader instances
            test_dataset = FeatureDataset(
                instance_information=test_dataset,
                label_mapping=settings["label_mapping"],
                iterations=None,
                feature_dropout_prob=0.0,
                maximum_features=25000,
            )
            test_dataloader = DataLoader(
                dataset=test_dataset,
                sampler= SequentialSampler(test_dataset),
                batch_size=1,
                num_workers=settings['training_config']['num_workers'],
                pin_memory=settings['training_config']['pin_memory'],
            )

            # record number of training and validation cases to check correctness
            if settings["level"] == "specimen":
                logger.info(f'Specimens: {len(test_dataset.feature_names)}')
            elif settings["level"] == "slide":
                logger.info(f'Slides: {len(test_dataset.feature_names)}')
            elif settings["level"] == "section":
                logger.info(f'Sections: {len(test_dataset.feature_names)}')
            else:
                raise NotImplementedError

            # select model type
            if settings['model'] == 'ViT':
                network = ViT
            else:
                ValueError(f'Invalid model name: {settings["model"]}')
            
            # initialize model
            model = network(**settings['model_config']['model_arguments'])
            
            # load checkpoint
            checkpoints_directory = experiments_directory/experiment_folder/'checkpoints'
            last_checkpoint = sorted(os.listdir(checkpoints_directory))[-1]
            checkpoint = torch.load(checkpoints_directory/last_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

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

            # define progress bar if specified
            settings["progress_bar"] = True
            progress = lambda x: (tqdm(x, total=len(test_dataset)) 
                                    if settings["progress_bar"] else x)

            # define labels
            label_indices = list(range(len(set(settings["label_mapping"].values()))))
            label_names = ['ALK', 'ROS1', 'NTRK'] + (['HRAS', 'BRAF', 'MAP3K8'] if len(label_indices) == 6 else ['other'])

            specimen_indices = []
            y_pred_prob = []
            y_pred_indices = []      
            y_true_indices = []     
            # deactivate autograd engine (backpropagation not required here)
            with torch.no_grad():
                for i, batch in progress(enumerate(test_dataloader)):
                    # get the specimen index and filenames
                    specimen_indices.append(int(batch[0]))
                    filenames = batch[-1]
                    batch = batch[1:-1]

                    # bring the data to the correct device
                    batch = [item.to(device) for item in batch]
                    x, pos, y, weight = batch

                    # get model prediction
                    output = model(x, pos, return_last_self_attention=save_attention)
                    if save_attention:
                        logit, attn, selected_pos = output
                    else:
                        logit = output

                    y_pred = F.softmax(logit, dim=1)
                    y_pred_prob.append(y_pred.tolist()[0])

                    y_pred_index = int(torch.argmax(y_pred))
                    y_true_index = int(torch.argmax(y))

                    y_pred_indices.append(y_pred_index)
                    y_true_indices.append(y_true_index)

                    if save_attention:
                        torch.save({
                            'attn': attn[0, :, 0, 1:].to('cpu'), 
                            'pos': selected_pos.to('cpu'), 
                            'filenames': filenames,
                        },
                            attn_directory/f'{specimen_indices[-1]}.pth',
                        )

            # calculate accuracy
            results_dict = {
                'accuracy': [accuracy_score(y_true_indices, y_pred_indices)],
                'balanced_accuracy': [balanced_accuracy_score(y_true_indices, y_pred_indices)],
            }
            if config['bootstrap']:
                acc_bootstrap_output, acc_sample_results = bootstrapper(
                    y_true=y_true_indices,
                    y_pred=y_pred_indices,
                    eval_func=accuracy_score,
                    iterations=iterations,
                    stratified=stratified,
                    confidence_level=confidence_level,
                    seed=0,
                    show_progress=verbose,
                    return_sample_results=save_sample_results,
                )
                balanced_acc_bootstrap_output, balanced_acc_sample_results = bootstrapper(
                    y_true=y_true_indices,
                    y_pred=y_pred_indices,
                    eval_func=balanced_accuracy_score,
                    iterations=iterations,
                    stratified=stratified,
                    confidence_level=confidence_level,
                    seed=0,
                    show_progress=verbose,
                    return_sample_results=save_sample_results,
                )
                acc_bootstrap_output = {f'accuracy_{k}':[v] for k,v in acc_bootstrap_output.items()}
                balanced_acc_bootstrap_output = {f'balanced_accuracy_{k}':[v] for k,v in balanced_acc_bootstrap_output.items()}
                bootstrap_dict = {**acc_bootstrap_output, **balanced_acc_bootstrap_output}

            # calculate AUROC
            avg_auroc = []
            sample_auroc = []
            for label_index, label_name in zip(list(range(len(label_names))), label_names):
                y_true = [1 if i == label_index else 0 for i in y_true_indices]
                y_pred = [y_pred[label_index] for y_pred in y_pred_prob]
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                auroc = roc_auc_score(y_true, y_pred)
                results_dict[f'{label_name}_vs_rest_auroc'] = [auroc]
                avg_auroc.append(auroc)
                if config['bootstrap']:
                    bootstrap_output, sample_results = bootstrapper(
                        y_true=y_true,
                        y_pred=y_pred,
                        eval_func=roc_auc_score,
                        iterations=iterations,
                        stratified=stratified,
                        confidence_level=confidence_level,
                        seed=0,
                        show_progress=verbose,
                        return_sample_results=True,
                    )
                    bootstrap_output = {f'{label_name}_vs_rest_{k}':[v] for k,v in bootstrap_output.items()}
                    bootstrap_dict = {**bootstrap_dict, **bootstrap_output}
                    sample_auroc.append(sample_results)
            
            results_dict[f'avg_one_vs_rest_auroc'] = [mean(avg_auroc)]
            if config['bootstrap']:
                sample_avg = []
                iterator = tqdm(range(iterations)) if verbose else range(iterations)
                for i in iterator:
                    sample_avg.append(mean([comparison[i] for comparison in sample_auroc]))
                bootstrap_results = {
                    'mean_avg_one_vs_rest_auroc': float(np.mean(sample_avg)),
                    f'95% CI lower': float(np.quantile(sample_avg, (1-confidence_level)/2)),
                    f'95% CI upper': float(np.quantile(sample_avg, 1-((1-confidence_level)/2))),
                }
                bootstrap_dict = {**bootstrap_dict, **bootstrap_results}

            if config['bootstrap'] and save_sample_results:
                df_acc_sample = pd.DataFrame.from_dict({'bootstrap samples': acc_sample_results})
                df_balanced_acc_sample = pd.DataFrame.from_dict({'bootstrap samples': balanced_acc_sample_results})
                df_avg_auroc_sample = pd.DataFrame.from_dict({'bootstrap samples': sample_avg})

            # create confusion matrix          
            disp = ConfusionMatrixDisplay.from_predictions(
                y_true=y_true_indices,
                y_pred=y_pred_indices,
                display_labels=label_names,
                cmap=plt.cm.Blues,
            )
            plt.savefig(results_directory/'confusion_matrix.eps')
            plt.savefig(results_directory/'confusion_matrix.jpg')
            plt.close()

            # convert dictionaries to dataframes
            predictions_df = pd.DataFrame.from_dict({
                "specimen_indices": specimen_indices,
                "y_pred_prob": y_pred_prob,
                "y_pred_indices": y_pred_indices,
                "y_true_indices": y_true_indices,
            })
            results_df = pd.DataFrame.from_dict(results_dict)
            if config['bootstrap']:
                bootstrap_df = pd.DataFrame.from_dict(bootstrap_dict)

            # save evaluation results in spreadsheet
            with pd.ExcelWriter(results_directory / 'results.xlsx') as writer:
                predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
                results_df.to_excel(writer, sheet_name='Results', index=False)
                if config['bootstrap']:
                    bootstrap_df.to_excel(writer, sheet_name='Bootrapped results', index=False)
                    if save_sample_results:
                        df_acc_sample.to_excel(writer, sheet_name='accuracy bootstrap samples', index=False)
                        df_balanced_acc_sample.to_excel(writer, sheet_name='balanced accuracy bootstrap samples', index=False)
                        df_avg_auroc_sample.to_excel(writer, sheet_name='avg AUROC bootstrap samples', index=False)