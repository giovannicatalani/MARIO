import numpy as np
import torch
from torch_geometric.data import Dataset, Data


class SDFDataset:
    def __init__(self, data_list, data_names, is_train=True, coef_norm=None, num_points=None):
        """
        Initialize the AirfransDataset class with pre-loaded data.
        
        Args:
            data_list: list of np.arrays containing the raw data
            data_names: list of data names
            target_field: str, one of ['velocity', 'pressure', 'turbulent_viscosity', 'implicit_distance']
            is_train: bool, whether this is training data (to compute normalization)
            coef_norm: dict, normalization coefficients (required if is_train=False)
            num_points: int, number of points to subsample (if None, use all points)
        """
        self.data_list = data_list
        self.data_names = data_names
        self.num_points = num_points
        self.is_train = is_train
        
        
        if is_train:
            # Compute normalization parameters from data
            print("Computing normalization parameters from training data...")
            self.compute_norm_params(data_list)
        else:
            # Use provided normalization parameters
            if coef_norm is None:
                raise ValueError("coef_norm must be provided when is_train=False")
            print("Using provided normalization parameters...")
            self.coef_norm = coef_norm
            self.pos_norm = coef_norm['pos_norm']

        # Process data
        print("Processing dataset...")
        self.processed_dataset = self.process_data(data_list)

    def compute_norm_params(self, data_list):
        """Compute normalization parameters."""
        self.coef_norm = {}
        
        all_targets = np.concatenate([data[:, 4:5] for data in data_list])
        
        self.coef_norm['mean'] = all_targets.mean(axis=0)
        self.coef_norm['std'] = all_targets.std(axis=0)
        
        # Calculate position normalization
        all_positions = np.concatenate([data[:, :2] for data in data_list])
        self.pos_norm = {
            'min': all_positions.min(axis=0),
            'max': all_positions.max(axis=0)
        }
        self.coef_norm['pos_norm'] = self.pos_norm
        
        print("\nNormalization parameters computed:")
        print(f"Target mean: {np.array2string(self.coef_norm['mean'], precision=4, separator=', ')}")
        print(f"Target std: {np.array2string(self.coef_norm['std'], precision=4, separator=', ')}")
        return self.coef_norm

    def process_data(self, dataset_list):
        """Process the raw data into the required format using global normalization."""
        dataset = []
    
        target_idx = 4
        
        # Process each simulation
        for i,sim_data in enumerate(dataset_list):
              
            # Extract and normalize positions
            pos = sim_data[:, :2]
            pos = 2 * (pos - self.pos_norm['min']) / (self.pos_norm['max'] - self.pos_norm['min']) - 1
            input_data = pos
            output = sim_data[:, target_idx:target_idx+1]
            
            # Normalize target using mean and std
            output = (output - self.coef_norm['mean']) / self.coef_norm['std']
            
            # Create data entry
            data_entry = Data(
                pos=torch.tensor(pos, dtype=torch.float),
                input=torch.tensor(input_data, dtype=torch.float),
                output=torch.tensor(output, dtype=torch.float),
                is_airfoil=torch.tensor(sim_data[:, -1]),
            )
            
            dataset.append(data_entry)
        
        # Subsample if requested
        if self.num_points is not None:
            dataset = self.subsample_dataset(dataset, self.num_points)
            
        return dataset

    @staticmethod
    def subsample_dataset(dataset, num_points):
        """Subsample the dataset while maintaining PyG Data format."""
        subsampled_dataset = []
        for data in dataset:
            total_points = data.num_nodes
            
            if total_points > num_points:
                subsample_indices = np.random.choice(total_points, num_points, replace=False)
                
                new_data = Data(
                    input=data.input[subsample_indices],
                    pos=data.pos[subsample_indices],
                    output=data.output[subsample_indices],
                    is_airfoil=data.is_airfoil[subsample_indices],
                )
                
                subsampled_dataset.append(new_data)
            else:
                subsampled_dataset.append(data)
                
        return subsampled_dataset
    
def subsample_dataset(dataset, num_points):
        """Subsample the dataset while maintaining PyG Data format."""
        subsampled_dataset = []
        for data in dataset:
            total_points = data.num_nodes
            
            if total_points > num_points:
                subsample_indices = np.random.choice(total_points, num_points, replace=False)
                
                new_data = Data(
                    input=data.input[subsample_indices],
                    pos=data.pos[subsample_indices],
                    output=data.output[subsample_indices],
                    is_airfoil=data.is_airfoil[subsample_indices],
                )
                
                subsampled_dataset.append(new_data)
            else:
                subsampled_dataset.append(data)
                
        return subsampled_dataset

import numpy as np
import torch
from torch_geometric.data import Dataset, Data

import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from typing import Literal


def make_bl_layer(sample, thickness: float = 0.03, decay: Literal['linear', 'exponential', 'squared'] = 'squared'):
    """
    Compute a boundary-layer mask based on the normalized inverse of the SDF field.

    Args:
        sample: dict with key 'point_cloud_field/distance_function' (1D np.ndarray)
        thickness: relative thickness (0 < thickness <= 1)
        decay: 'linear', 'exponential', or 'squared'
    Returns:
        np.ndarray of shape (N,), boundary-layer mask in [0, 1]
    """
    if not (0 < thickness <= 1):
        raise ValueError('Thickness should be in (0, 1].')

    def threshold(var, value):
        mask = var > value
        var = var.copy()
        var[~mask] = 0
        if mask.any():
            var[mask] = (var[mask] - value) / (var[mask].max() - value)
        return var

    df = sample['point_cloud_field/distance_function'].max() - sample['point_cloud_field/distance_function']
    df = df / df.max()

    if decay == 'squared':
        return threshold(df, 1 - thickness) ** 2
    elif decay == 'linear':
        return threshold(df, 1 - thickness)
    elif decay == 'exponential':
        return (np.exp(threshold(df, 1 - thickness)) - 1) / (np.exp(1) - 1)
    else:
        raise ValueError(f"Unknown decay type: {decay}")


class AirfransFlowDataset(Dataset):
    """
    PyG Dataset for Airfrans flow data.

    Modes:
      - 'train': compute normalization stats
      - 'val'/'test': reuse train stats

    Each point in the dataset is described by:
      - Position (x, y): indices [0, 1]
      - Inlet velocity (u_in, v_in): indices [2, 3]
      - Distance to airfoil (SDF): index [4]
      - Normals (nx, ny): indices [5, 6]
      - Airfoil boolean: index [11]

    Output fields (targets):
      - Velocity (u, v): indices [7, 8]
      - Pressure/density: [9]
      - Turbulent viscosity: [10]

    Input normalization:
      - [x, y], sdf, normals, bl_mask → min-max normalized to [-1, 1]
    Output normalization:
      - [u, v, p, nu_t] → standard (mean/std)
    Conditional normalization:
      - [geom_latents, inlet_velocity] → standard (mean/std)
    """

    def __init__(
        self,
        data_list,
        data_names,
        geom_latents,
        mode='train',
        coef_norm=None,
        num_points=None,
        idx_list=None,
        transform=None,
        pre_transform=None
    ):
        super().__init__(None, transform, pre_transform)
        self.data_list = data_list
        self.data_names = data_names
        self._raw_geom_latents = np.array(geom_latents)
        
        outlier_names = ["airFoil2D_SST_50.077_-4.416_2.834_4.029_1.0_5.156"]
        to_remove = [i for i, n in enumerate(self.data_names) if n in outlier_names]
        if to_remove:
            print(f"Removing outlier(s): {[self.data_names[i] for i in to_remove]}")
            self.data_list = [d for i, d in enumerate(self.data_list) if i not in to_remove]
            self.data_names = [n for i, n in enumerate(self.data_names) if i not in to_remove]
            self._raw_geom_latents = np.delete(self._raw_geom_latents, to_remove, axis=0)
        
        total = len(self.data_list)
        self.idx_list = list(range(total)) if idx_list is None else list(idx_list)
        self.geom_latents = self._raw_geom_latents[self.idx_list]
        self.mode = mode
        self.num_points = num_points

        # Define indices
        self.pos_indices = [0, 1]               # Position (x, y)
        self.inlet_vel_indices = [2, 3]         # Inlet velocity
        self.sdf_index = 4                      # Distance to airfoil
        self.normal_indices = [5, 6]            # Normal components
        self.output_indices = [7, 8, 9, 10]     # Output fields
        self.airfoil_index = 11                 # Airfoil boolean

        # Boundary layer mask parameters
        self.bl_thickness = 0.03
        self.bl_decay = 'squared'

        # Output field names
        self.output_fields = ['Velocity-x', 'Velocity-y', 'Pressure', 'Turbulent-viscosity']

        # Normalization coefficients
        if mode == 'train':
            self.coef_norm = self.compute_norm_params()
        else:
            if coef_norm is None:
                raise ValueError("coef_norm required for non-train modes")
            self.coef_norm = coef_norm

        # Process dataset
        self.data_list = self.process_dataset()

    def compute_norm_params(self):
        """Compute normalization parameters from training subset."""
        pos_all, sdf_all, normal_all, bl_all = [], [], [], []
        output_fields_all = {i: [] for i in range(len(self.output_indices))}
        cond_all = []

        for idx in self.idx_list:
            data = self.data_list[idx]
            pos = data[:, self.pos_indices]
            sdf = data[:, self.sdf_index]
            normals = data[:, self.normal_indices]
            outputs = data[:, self.output_indices]
            inlet_vel = data[:, self.inlet_vel_indices]

            # Boundary layer mask
            sample = {'point_cloud_field/distance_function': sdf}
            bl_mask = make_bl_layer(sample, thickness=self.bl_thickness, decay=self.bl_decay)

            pos_all.append(pos)
            sdf_all.append(sdf)
            normal_all.append(normals)
            bl_all.append(bl_mask)

            # Outputs
            for i in range(len(self.output_indices)):
                output_fields_all[i].append(outputs[:, i])

            # Conditional latent
            cond = np.concatenate([
                self.geom_latents[self.idx_list.index(idx)],
                np.mean(inlet_vel, axis=0)
            ])
            cond_all.append(cond)

        # Position
        pos_all = np.vstack(pos_all)
        pos_min = pos_all.min(axis=0)
        pos_max = pos_all.max(axis=0)
        pos_range = pos_max - pos_min
        pos_range[pos_range == 0] = 1.0

        # SDF
        sdf_all = np.hstack(sdf_all)
        sdf_min, sdf_max = sdf_all.min(), sdf_all.max()
        sdf_range = sdf_max - sdf_min
        sdf_range = 1.0 if sdf_range == 0 else sdf_range

        # Normals
        normal_all = np.vstack(normal_all)
        normal_min = normal_all.min(axis=0)
        normal_max = normal_all.max(axis=0)
        normal_range = normal_max - normal_min
        normal_range[normal_range == 0] = 1.0

        # BL mask
        bl_all = np.hstack(bl_all)
        bl_min, bl_max = bl_all.min(), bl_all.max()
        bl_range = bl_max - bl_min
        bl_range = 1.0 if bl_range == 0 else bl_range

        # Outputs
        output_means, output_stds = {}, {}
        for i, field_name in enumerate(self.output_fields):
            field_data = np.hstack(output_fields_all[i])
            output_means[field_name] = float(field_data.mean())
            output_stds[field_name] = float(field_data.std()) or 1.0

        # Conditional inputs
        cond_all = np.vstack(cond_all)
        cond_mean = cond_all.mean(axis=0)
        cond_std = cond_all.std(axis=0)
        cond_std[cond_std == 0] = 1.0

        return {
            'pos': {'min': pos_min, 'max': pos_max, 'range': pos_range},
            'sdf': {'min': sdf_min, 'max': sdf_max, 'range': sdf_range},
            'normal': {'min': normal_min, 'max': normal_max, 'range': normal_range},
            'bl': {'min': bl_min, 'max': bl_max, 'range': bl_range},
            'output': {'mean': output_means, 'std': output_stds},
            'cond': {'mean': cond_mean, 'std': cond_std}
        }

    def process_dataset(self):
        """Convert dataset into PyG Data objects."""
        processed_data_list = []
        c = self.coef_norm

        for i, idx in enumerate(self.idx_list):
            data = self.data_list[idx]

            pos = data[:, self.pos_indices]
            sdf = data[:, self.sdf_index]
            normals = data[:, self.normal_indices]
            airfoil_bool = data[:, self.airfoil_index]

            # Subsample if requested
            if self.num_points and pos.shape[0] > self.num_points:
                sel = np.random.choice(pos.shape[0], self.num_points, replace=False)
                pos, sdf, normals, airfoil_bool = pos[sel], sdf[sel], normals[sel], airfoil_bool[sel]
                data = data[sel]

            # Compute BL mask
            sample = {'point_cloud_field/distance_function': sdf}
            bl_mask = make_bl_layer(sample, thickness=self.bl_thickness, decay=self.bl_decay)

            # Normalize all inputs
            pos_n = 2 * (pos - c['pos']['min']) / c['pos']['range'] - 1
            sdf_n = 2 * ((sdf - c['sdf']['min']) / c['sdf']['range']) - 1
            normals_n = 2 * (normals - c['normal']['min']) / c['normal']['range'] - 1
            bl_n = 2 * ((bl_mask - c['bl']['min']) / c['bl']['range']) - 1

            # Inputs: [x, y, sdf, nx, ny, bl_mask]
            input_feats = np.concatenate([pos_n, sdf_n[:, None], normals_n, bl_n[:, None]], axis=1)
            node_kwargs = {
                'input': torch.tensor(input_feats, dtype=torch.float),
                'pos': torch.tensor(pos_n, dtype=torch.float),
                'is_airfoil': torch.tensor(airfoil_bool, dtype=torch.float)
            }

            # Outputs
            output_data = []
            for j, field_name in enumerate(self.output_fields):
                field_data = data[:, self.output_indices[j]]
                mean_val = c['output']['mean'][field_name]
                std_val = c['output']['std'][field_name]
                field_norm = (field_data - mean_val) / std_val
                output_data.append(field_norm[:, None])
            node_kwargs['output'] = torch.tensor(np.hstack(output_data), dtype=torch.float)

            # Conditional inputs
            inlet_vel = np.mean(data[:, self.inlet_vel_indices], axis=0)
            cond = np.concatenate([self.geom_latents[i], inlet_vel])
            cm, cs = c['cond']['mean'], c['cond']['std']
            cond_n = (cond - cm) / cs
            node_kwargs['cond'] = torch.tensor(cond_n, dtype=torch.float).unsqueeze(0)

            processed_data_list.append(Data(**node_kwargs))

        return processed_data_list

    def split_val(self, val_size, seed=None):
        rng = np.random.RandomState(seed)
        chosen = rng.choice(self.idx_list, size=val_size, replace=False).tolist()
        train_idx = [i for i in self.idx_list if i not in chosen]

        train_ds = AirfransFlowDataset(
            self.data_list,
            self.data_names,
            self._raw_geom_latents,
            mode='train',
            num_points=self.num_points,
            idx_list=train_idx
        )

        val_ds = AirfransFlowDataset(
            self.data_list,
            self.data_names,
            self._raw_geom_latents,
            mode='val',
            coef_norm=train_ds.coef_norm,
            num_points=self.num_points,
            idx_list=chosen
        )

        return train_ds, val_ds

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]



def subsample_dataset(dataset, num_points, seed=None):
    """
    Subsample each graph in `dataset` to at most `num_points` nodes,
    by randomly selecting rows from `data.input` and `data.output`.
    
    Args:
        dataset: iterable of torch_geometric.data.Data
        num_points: int, max number of nodes per graph
        seed: optional int, for reproducibility
        
    Returns:
        List[Data]: new list of Data objects with subsampled inputs/outputs
    """
    if seed is not None:
        np.random.seed(seed)

    out = []
    for data in dataset:
        N = data.input.size(0)
        if N > num_points:
            # pick random subset of node indices
            idx = np.random.choice(N, num_points, replace=False)
            idx = torch.from_numpy(idx).long()

            # subsample inputs
            inp_sub = data.input[idx]
            pos_sub = data.pos[idx] 
            # subsample outputs if they exist
            out_sub = data.output[idx] if hasattr(data, 'output') else None

            # build new Data
            new_data = Data(input=inp_sub, pos=pos_sub)
            if out_sub is not None:
                new_data.output = out_sub
            # carry through the graph-level fields unchanged
            if hasattr(data, 'cond'):
                new_data.cond = data.cond
            if hasattr(data, 'output_scalars'):
                new_data.output_scalars = data.output_scalars
        else:
            # nothing to do
            new_data = data

        out.append(new_data)
    return out

from typing import Literal

def make_bl_layer(sample, thickness: float = 0.03, decay: Literal['linear', 'exponential', 'squared'] = 'squared'):
    """
    Compute a boundary-layer mask based on the normalized inverse of the SDF field.

    Args:
        sample: dict or object with key 'point_cloud_field/distance_function' (1D np.ndarray)
        thickness: relative thickness (0 < thickness <= 1)
        decay: 'linear', 'exponential', or 'squared'
    Returns:
        np.ndarray of shape (N,), boundary-layer mask in [0, 1]
    """
    if not (0 < thickness <= 1):
        raise ValueError('Thickness should be in (0, 1].')

    def threshold(var, value):
        mask = var > value
        var = var.copy()
        var[~mask] = 0
        if mask.any():
            var[mask] = (var[mask] - value) / (var[mask].max() - value)
        return var

    df = sample['point_cloud_field/distance_function'].max() - sample['point_cloud_field/distance_function']
    df = df / df.max()

    if decay == 'squared':
        return threshold(df, 1 - thickness) ** 2
    elif decay == 'linear':
        return threshold(df, 1 - thickness)
    elif decay == 'exponential':
        return (np.exp(threshold(df, 1 - thickness)) - 1) / (np.exp(1) - 1)
    else:
        raise ValueError(f"Unknown decay type: {decay}")