from typing import List
import numpy as np
import torch

from torch.utils.data import ConcatDataset

__all__ = ["ConcatDataset"]


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets, dataset_ids: List[int] = None):
        for i in range(len(datasets)):
            if callable(datasets[i]):
                datasets[i] = datasets[i]()

        if dataset_ids is not None:
            datasets = [ds for ds in datasets if ds.dataset_id in dataset_ids]

        super().__init__(datasets)
        

    def _calculate_weights(self, datasets):
        # Calculate weights for each sample in each dataset
        dataset_lengths = np.array([len(ds) for ds in datasets])
        weight = 1.0 / dataset_lengths
        # Create a list of weights
        weights = np.repeat(weight, dataset_lengths)
        return weights
    
    def _calculate_label_weights(self, datasets):
        # Concatenate all labels from all datasets
        all_labels = torch.cat([torch.tensor(ds.get_labels()) for ds in datasets])
    
        # Count the number of occurrences of each class
        class_counts = torch.bincount(all_labels)
        
        # Calculate weights
        weights = 1. / class_counts[all_labels]
        
        return weights

    def get_weights(self):
        # Return the calculated weights
        return self._calculate_weights(self.datasets)
    
    def get_label_weights(self):
        return self._calculate_label_weights(self.datasets)
