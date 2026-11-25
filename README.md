<div align="center">

# Generalization of Artificial Intelligence Models for Predicting Genetic Driver Alterations in Spitz Tumors

Ruben T. Lucassen, Chiel F. Ebbelaar, Gerben E. Breimer, Jean-Yves Blay, Anne M. L. Jansen, Mitko Veta, Arnaud de la Fouchardière & Willeke A. M. Blokx

*The paper is currently under review*

</div>

## Contents
The repository contains the following folders and files:
- `preprocessing` contains the scripts for tissue segmentation and correction, cross-section extraction, rotation prediction and correction, tessellation, stain extraction and clustering, feature extraction, dataset splitting, and summarizing the dataset characteristics (numbered in order of execution).
- `model` contains the implementation of the ViT aggregation model.
- `training.py` and `training_utils.py` contain the model training loop and the supporting utilities, such as the feature dataset.
- `evaluation.py` contains the script for inference and evaluation of individual models.
- `ensemble_evaluation.py` contains the script for evaluation of the model ensemble.
- `requirements.txt` lists the required Python packages.
