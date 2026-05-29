"""
Evaluate model ensemble.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
    roc_curve, 
    roc_auc_score, 
)
from statistics import mean
from tqdm import tqdm

from evaluation import bootstrapper


experiment_folders_path = Path(r"")
folder_dict = {int(folder.split('_')[0]): folder for folder in os.listdir(experiment_folders_path) if '-' not in folder.split('_')[0]}
iterators = []
experiment_ensembles = [[folder_dict[i] for i in iterator] for iterator in iterators]

bootstrap = True
confidence_level = 0.95
iterations = 10000
stratified = True
verbose = True
overwrite = False
save_sample_results = True


if __name__ == '__main__':

    for folders in experiment_ensembles:
        indices = [int(name.split('_')[0]) for name in folders]
        ensemble_name = f"{min(indices)}-{max(indices)}_ensemble_{'_'.join(folders[0].split('_')[1:])}"
        print(ensemble_name)

        subfolder_dict = {}
        for folder in folders:
            for item in os.listdir(experiment_folders_path/folder):
                if os.path.isdir(experiment_folders_path/folder/item) and 'results' in item:
                    if item not in subfolder_dict:
                        subfolder_dict[item] = [folder]
                    else:
                        subfolder_dict[item].append(folder)
        
        for subfolder in subfolder_dict:
            print(subfolder)
            if subfolder.endswith('same_validation'):
                continue

            if len(subfolder_dict[subfolder]) != len(folders):
                print('Skipped because subfolder was not present in all folders')
                continue

            y_pred_probs = []
            y_true_indices = []
            specimen_indices = []
            if len(subfolder_dict[subfolder]) == len(folders):
                for folder in folders:
                    results_path = experiment_folders_path/folder/subfolder/'results.xlsx'
                    df = pd.read_excel(results_path)
                    y_pred_probs.append([eval(i) for i in df['y_pred_prob']])
                    y_true_indices.append(df['y_true_indices'].tolist())
                    specimen_indices.append(df['specimen_indices'].tolist())

            y_pred_probs = torch.tensor(y_pred_probs)
            y_pred_prob = torch.mean(y_pred_probs, dim=0)
            y_pred_indices = torch.argmax(y_pred_prob, dim=1)

            results_directory = experiment_folders_path/ensemble_name/subfolder
            if os.path.exists(results_directory):
                if not overwrite:
                    continue
            else:
                os.makedirs(results_directory)

            if all([y_true_indices[0] == y_true_indices[i] for i in range(1, len(y_true_indices))]):
                y_true_indices = y_true_indices[0]
            else:
                raise ValueError
            
            if all([specimen_indices[0] == specimen_indices[i] for i in range(1, len(specimen_indices))]):
                specimen_indices = specimen_indices[0]
            else:
                raise ValueError

            # define labels
            label_names = ['ALK', 'ROS1', 'NTRK'] + (['HRAS', 'BRAF', 'MAP3K8'] if len(y_pred_prob[0, :]) == 6 else ['other'])

            # calculate accuracy
            results_dict = {
                'accuracy': [accuracy_score(y_true_indices, y_pred_indices)],
                'balanced_accuracy': [balanced_accuracy_score(y_true_indices, y_pred_indices)],
            }
            if bootstrap:
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
                if bootstrap:
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
            if bootstrap:
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
                "y_pred_prob": y_pred_prob.tolist(),
                "y_pred_indices": y_pred_indices,
                "y_true_indices": y_true_indices,
            })
            results_df = pd.DataFrame.from_dict(results_dict)
            if bootstrap:
                bootstrap_df = pd.DataFrame.from_dict(bootstrap_dict)
                if save_sample_results:
                    df_acc_sample = pd.DataFrame.from_dict({'bootstrap samples': acc_sample_results})
                    df_balanced_acc_sample = pd.DataFrame.from_dict({'bootstrap samples': balanced_acc_sample_results})
                    df_avg_auroc_sample = pd.DataFrame.from_dict({'bootstrap samples': sample_avg})

            # save evaluation results in spreadsheet
            with pd.ExcelWriter(results_directory / 'results.xlsx') as writer:
                results_df.to_excel(writer, sheet_name='Results', index=False)
                if bootstrap:
                    bootstrap_df.to_excel(writer, sheet_name='Bootrapped results', index=False)
                    if save_sample_results:
                        df_acc_sample.to_excel(writer, sheet_name='accuracy bootstrap samples', index=False)
                        df_balanced_acc_sample.to_excel(writer, sheet_name='balanced acc bootstrap samples', index=False)
                        df_avg_auroc_sample.to_excel(writer, sheet_name='avg AUROC bootstrap samples', index=False)
                predictions_df.to_excel(writer, sheet_name='Predictions', index=False)