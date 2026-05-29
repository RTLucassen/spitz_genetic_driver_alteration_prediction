"""
Utility functions for model training.
"""

import random
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        instance_information: dict[str, dict[str, Any]],
        label_mapping: dict[str, int],
        iterations: Optional[int] = None,
        feature_dropout_prob: float = 0.0,
        maximum_features: Optional[int] = None,
    ) -> None:
        """
        Initializes dataset instance to load feature vectors.

        Args:
            instance_information:  Dictionary with information about the instance (i.e., specimen or section)
            label_mapping:  Dictionary with label names as keys and indices for one-hot encoding as values.
            iterations:  Number of iterations for training.
            feature_dropout_prob:  Probability for randomly dropping out a feature.
            maximum_features:  Maximum number of features. If more features are available, 
                randomly select a subset equal to the maximum.
        """
        # define instance attributes
        self.instance_information = instance_information
        self.label_mapping = label_mapping
        self.feature_names = list(self.instance_information.keys())
        self.iterations = iterations
        self.feature_dropout_prob = feature_dropout_prob
        self.maximum_features = maximum_features

    def __len__(self) -> int:
        if self.iterations is None:
            return len(self.feature_names)
        else:
            return self.iterations

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
        """
        Load one or more sets of features for the instance.

        Args:
            index:  Index for selecting features from the dataset.

        Returns:
            specimen_index:  Index of specimen in the dataset.
            features:  Concatenated feature vectors.
            positions:  Contatenated position vectors corresponding 
                to the feature vectors.
            label:  Specimen label.
            weight:  Instance weight for balancing loss contribution.
        """
        # get the instance index
        instance_idx = self.feature_names[index % len(self.feature_names)]
        # get instance information
        instance = self.instance_information[instance_idx]

        features = []
        positions = []
        # load feature dictionary
        for i, feature_name in enumerate(instance['filenames']):
            feature_dict = torch.load(Path(instance['directory'])/feature_name) 
            features.append(feature_dict['feature'])
            # used for generating attention maps
            if True:
                positions.append(torch.concat([
                    feature_dict['position'], 
                    torch.ones((feature_dict['position'].shape[0], 1))*i,
                    torch.tensor(list(range(feature_dict['position'].shape[0])))[:, None]
                ], dim=1))
            else:
                positions.append(feature_dict['position'])
            # check correctness of label and convert to one-hot encoding
            if feature_dict['label'] != instance['label']:
                raise ValueError(f"{feature_dict['label']} and {instance['label']} not not match")
        
        # concatenate features
        features = torch.concatenate(features, dim=0)
        positions = torch.concatenate(positions, dim=0)
        
        # create one-hot encoding of the label
        label = F.one_hot(
            torch.tensor(self.label_mapping[instance['label']]), 
            num_classes=len(set(self.label_mapping.values())),
        )
        # get features and randomly drop out features
        if self.feature_dropout_prob > 0.0:
            # create the selection mask
            selection_mask = torch.rand(features.shape[0]) > self.feature_dropout_prob 
            # include a random feature if none were selected
            if not selection_mask.any():
                selection_mask = [False]*features.shape[0]
                selection_mask[random.randint(0, len(selection_mask)-1)] = True

            # apply the selection mask
            features = features[selection_mask, :]
            positions = positions[selection_mask, :]

        # randomly select features up to the maximum number
        if self.maximum_features is not None and features.shape[0] > self.maximum_features:
            indices = list(range(features.shape[0]))
            random.shuffle(indices)
            features = features[indices[:self.maximum_features], :]
            positions = positions[indices[:self.maximum_features], :]

        # convert features and positions to torch tensors
        features = features.to(torch.float32)
        positions = positions.to(torch.float32)

        return instance['specimen_index'], features, positions, label, instance['weight'], instance['filenames']


class FocalLoss(nn.Module):

    def __init__(
        self,
        sigmoid: bool = False, 
        gamma: float = 0.0,
        class_weights: Optional[Union[list[float], torch.Tensor]] = None,
    ) -> None:
        """
        Initialize focal loss.

        Args:
            sigmoid:  Specify if a sigmoid instead of a softmax function is applied.
                If there is only a single class, the sigmoid is automatically used.
            gamma:  Parameter that governs the relative importance of incorrect 
                predictions. If gamma equals 0.0, the focal loss is equal to the 
                cross-entropy loss.
            class_weights:  If not None, compute a weighted average of the loss 
                for the classes. 
        """
        super().__init__()
        self.sigmoid = sigmoid
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logit: torch.Tensor, y_true: torch.Tensor, 
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ 
        Args:
            logit:  Logit predictions volumes of shape: (batch, class, X, Y, ...).
            y_true:  True label volumes of matching shape: (batch, class, X, Y, ...).
            weight:  Weighting factor for each item in the batch. 

        Returns:
            loss:  Focal loss averaged over all images in the batch.
        """
        # check if the logit prediction and true labels are of equal shape
        if logit.shape != y_true.shape:
            raise ValueError('Shape of predicted and true labels do not match.')
        
        # check if the values of y_true range between 0.0-1.0
        if torch.min(y_true) < 0.0 or torch.max(y_true) > 1.0:
            raise ValueError('Invalid values for y_true (outside the range 0.0-1.0).')
        
        # convert class weights from list to tensor if necessary
        if isinstance(self.class_weights, list):
            self.class_weights = torch.Tensor(self.class_weights)
        # check if the number of classes matches the number of class weights if provided
        if self.class_weights is not None:
            if len(self.class_weights) != logit.shape[1]:
                raise ValueError('The number of class weights and classes do not match.')
        else:
            self.class_weights = [1]*logit.shape[1]

        # get the predicted probabilities by taking
        # the sigmoid or softmax of the logit returned by the network
        if self.sigmoid or logit.shape[1] == 1:
            y_pred = torch.sigmoid(logit)
            log_y_pred = F.logsigmoid(logit)
        else:
            y_pred = torch.softmax(logit, dim=1)
            log_y_pred = F.log_softmax(logit, dim=1)

        # flatten the data (but keep the dimension of the batch and channels)
        y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
        y_pred_flat = y_pred.contiguous().view(*y_pred.shape[0:2], -1)
        log_y_pred_flat = log_y_pred.contiguous().view(*y_pred.shape[0:2], -1)

        # calculate the pixelwise cross-entropy, focal weight, and pixelwise focal loss
        pixelwise_CE = -(log_y_pred_flat * y_true_flat)
        focal_weight = (1-(y_true_flat * y_pred_flat))**self.gamma
        pixelwise_focal_loss = focal_weight * pixelwise_CE

        # calculate the class-separated focal loss
        class_separated_focal_loss = torch.mean(pixelwise_focal_loss, dim=-1)
        
        # multiply the loss per class by the class weight
        for i, class_weight in enumerate(self.class_weights):
            class_separated_focal_loss[:, i] *= class_weight
        instance_loss = torch.sum(class_separated_focal_loss, dim=1)

        # if a weight was specified, multiple each item in the batch with
        if weight is not None:
            instance_loss *= weight

        # compute the mean loss over the batch
        loss = torch.mean(instance_loss)

        return loss