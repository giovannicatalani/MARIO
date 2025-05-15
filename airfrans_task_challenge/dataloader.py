import numpy as np
import torch
from pyoche.utils.harmonizers import forced_to_fupc
from torch.utils.data import Dataset
from utils.callbacks import Callback

from torch_geometric.data import Data


class InrVupc(Dataset):
    def __init__(
        self, 
        pyoche_dataset, 
        input_features,
        output_features,
        conditioning_features,
        num_points=None
    ):
        self.num_points = num_points
        self.sample_n_points = []

        self.pyoche_dataset = pyoche_dataset
        self.input_features = input_features
        self.output_features = output_features
        self.conditioning_features = conditioning_features

        for sample in pyoche_dataset:
            self.sample_n_points.append(sum(sample[self.input_features[0]].shape[1:]))
        
        self.subsample()

    def __len__(self):
        return len(self.pyoche_dataset)
    
    def __getitem__(self, idx):
        sample = self.pyoche_dataset[idx]
        harmonizer = forced_to_fupc(self.sample_indices[idx].shape[0])

        x = torch.concat([torch.tensor(
            harmonizer(sample[input_feature].T[self.sample_indices[idx]]), 
            dtype=torch.float
        ) for input_feature in self.input_features], dim=1)

        y = torch.concat([torch.tensor(
            harmonizer(sample[output_feature].T[self.sample_indices[idx]]), 
            dtype=torch.float
        ) for output_feature in self.output_features], dim=1)


        z = torch.concat([
                torch.tensor(
                    harmonizer(sample[c_ft].T), dtype=torch.float
                ) 
                if sample[c_ft].ndim == 1 

                else torch.tensor(
                    harmonizer(sample[c_ft].T[self.sample_indices[idx]]), 
                    dtype=torch.float
                ) 

                for c_ft in self.conditioning_features
        ], dim=1)

        return Data(x = x, y = y, z = z)
    
    def get_full_item(self, idx):
        sample = self.pyoche_dataset[idx]
        harmonizer = forced_to_fupc(sample['point_cloud_field/is_surf'].shape[1])

        x = torch.concat([torch.tensor(
            harmonizer(sample[input_feature].T), 
            dtype=torch.float
        ) for input_feature in self.input_features], dim=1)

        y = torch.concat([torch.tensor(
            harmonizer(sample[output_feature].T), 
            dtype=torch.float
        ) for output_feature in self.output_features], dim=1)


        z = torch.concat([
                torch.tensor(
                    harmonizer(sample[c_ft].T), dtype=torch.float
                ) 
                if sample[c_ft].ndim == 1 

                else torch.tensor(
                    harmonizer(sample[c_ft].T), 
                    dtype=torch.float
                ) 

                for c_ft in self.conditioning_features
        ], dim=1)

        return Data(x = x, y = y, z = z, surf=sample['point_cloud_field/is_surf'].astype(bool)[0])
    
    def subsample(self):
        if self.num_points is None:
            self.sample_indices = [
                np.arange(snp) for snp in self.sample_n_points
            ]
            return
        
        self._generate_subsample_indices()
    
    def _generate_subsample_indices(self):

        self.sample_indices = []
        for snp in self.sample_n_points:
            sample_size = min(self.num_points, snp)
            sample_indices = np.random.choice(
                np.arange(0, snp),
                size=sample_size,
                replace=False
            )
            self.sample_indices.append(sample_indices)

class DloaderSubsample(Callback):
    def on_epoch_end(self, trainer, /, *args, **kwargs):
        trainer.train_dataloader.dataset.subsample()
        trainer.val_dataloader.dataset.subsample()


