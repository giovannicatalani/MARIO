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

class AirfransFlowDataset(Dataset):
    """
    PyG Dataset for Airfrans flow data.

    Modes:
      - 'train': include inputs, pos, outputs, cond; computes normals
      - 'val':   include inputs, pos, outputs, cond; uses provided coef_norm
      - 'test':  include inputs, pos, cond only; uses provided coef_norm

    Each point in the dataset is described by:
      - Position (2D): indices 0, 1
      - Inlet velocity (2D): indices 2, 3
      - Distance to airfoil (SDF): index 4
      - Normals (2D): indices 5, 6 (set to 0 if point is not on airfoil)
      
    Output fields (target):
      - Velocity (2D): indices 7, 8
      - Pressure/density: index 9
      - Turbulent kinematic viscosity: index 10
      
    Additional:
      - Airfoil boolean: index 11

    Normalizations:
      inputs: [x, y] & sdf -> min-max to [-1, 1]
      outputs: [velocity-x, velocity-y, pressure, turb_viscosity] -> standard
      cond: geom_latents + inlet_velocity -> standard
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
        total = len(data_list)
        self.idx_list = list(range(total)) if idx_list is None else list(idx_list)
        self.geom_latents = self._raw_geom_latents[self.idx_list]
        self.mode = mode
        self.num_points = num_points
        
        # Define indices
        self.pos_indices = [0, 1]               # Position (x, y)
        self.inlet_vel_indices = [2, 3]         # Inlet velocity components
        self.sdf_index = 4                      # Distance to airfoil
        self.normal_indices = [5, 6]            # Normal components
        self.output_indices = [7, 8, 9, 10]     # Output fields: velocity (2), pressure, viscosity
        self.airfoil_index = 11                 # Airfoil boolean
        
        # Output field names
        self.output_fields = ['Velocity-x', 'Velocity-y', 'Pressure', 'Turbulent-viscosity']
        
        # norms
        if mode == 'train':
            self.coef_norm = self.compute_norm_params()
        else:
            if coef_norm is None:
                raise ValueError("coef_norm required for non-train modes")
            self.coef_norm = coef_norm
        
        self.data_list = self.process_dataset()

    def compute_norm_params(self):
        """Compute normalization parameters from training data."""
        pos_all = []
        sdf_all = []
        output_fields_all = {i: [] for i in range(len(self.output_indices))}
        inlet_vel_all = []
        cond_all = []
        
        for idx in self.idx_list:
            data = self.data_list[idx]
            
            # Extract position, SDF, and outputs
            pos = data[:, self.pos_indices]
            sdf = data[:, self.sdf_index]
            outputs = data[:, self.output_indices]
            inlet_vel = data[:, self.inlet_vel_indices]
            
            pos_all.append(pos)
            sdf_all.append(sdf)
            
            if self.mode in ('train', 'val'):
                for i in range(len(self.output_indices)):
                    output_fields_all[i].append(outputs[:, i])
            
            # Collect inlet velocity for conditional input
            inlet_vel_all.append(np.mean(inlet_vel, axis=0))  # Use average inlet velocity
            
            # Add latent conditions with inlet velocities
            cond = np.concatenate([
                self.geom_latents[self.idx_list.index(idx)], 
                np.mean(inlet_vel, axis=0)  # Average inlet velocity
            ])
            cond_all.append(cond)
        
        # Process position data
        pos_all = np.vstack(pos_all)
        pos_min = pos_all.min(axis=0)
        pos_max = pos_all.max(axis=0)
        pos_range = pos_max - pos_min
        pos_range[pos_range == 0] = 1.0
        
        # Process SDF data
        sdf_all = np.hstack(sdf_all)
        sdf_min = sdf_all.min()
        sdf_max = sdf_all.max()
        sdf_range = sdf_max - sdf_min
        if sdf_range == 0:
            sdf_range = 1.0
        
        # Process output fields
        output_means = {}
        output_stds = {}
        
        if self.mode in ('train', 'val'):
            for i, field_name in enumerate(self.output_fields):
                field_data = np.hstack(output_fields_all[i])
                output_means[field_name] = float(field_data.mean())
                output_stds[field_name] = float(field_data.std())
                if output_stds[field_name] == 0:
                    output_stds[field_name] = 1.0
        
        # Process conditional inputs
        cond_all = np.vstack(cond_all)
        cond_mean = cond_all.mean(axis=0)
        cond_std = cond_all.std(axis=0)
        cond_std[cond_std == 0] = 1.0
        
        # Create and return normalization parameters
        return {
            'pos': {'min': pos_min, 'max': pos_max, 'range': pos_range},
            'sdf': {'min': sdf_min, 'max': sdf_max, 'range': sdf_range},
            'output': {'mean': output_means, 'std': output_stds},
            'cond': {'mean': cond_mean, 'std': cond_std}
        }

    def process_dataset(self):
        """Process the dataset into PyG Data objects."""
        processed_data_list = []
        c = self.coef_norm
        
        for i, idx in enumerate(self.idx_list):
            data = self.data_list[idx]
            
            # Extract required components
            pos = data[:, self.pos_indices]
            sdf = data[:, self.sdf_index]
            airfoil_bool = data[:, self.airfoil_index]
            
            # Subsample if requested
            if self.num_points and pos.shape[0] > self.num_points:
                sel = np.random.choice(pos.shape[0], self.num_points, replace=False)
                pos = pos[sel]
                sdf = sdf[sel]
                airfoil_bool = airfoil_bool[sel]
                # Also subsample outputs if needed
                if self.mode in ('train', 'val'):
                    data = data[sel]
            
            # Normalize position to [-1, 1]
            pmin, pmax = c['pos']['min'], c['pos']['max']
            prange = c['pos']['range']
            pos_n = 2 * (pos - pmin) / prange - 1
            
            # Normalize SDF to [-1, 1]
            smin, srange = c['sdf']['min'], c['sdf']['range']
            sdf_n = 2 * ((sdf - smin) / srange) - 1
            
            # Combine position and SDF as input features
            input_feats = torch.tensor(np.concatenate([pos_n, sdf_n[:, None]], axis=1), dtype=torch.float)
            
            # Prepare node kwargs
            node_kwargs = {
                'input': input_feats,
                'pos': torch.tensor(pos_n, dtype=torch.float),
                'is_airfoil': torch.tensor(airfoil_bool, dtype=torch.float)
            }
            
            # Add outputs if in train or val mode
            if self.mode in ('train', 'val'):
                output_data = []
                for i, field_name in enumerate(self.output_fields):
                    field_data = data[:, self.output_indices[i]]
                    
                    # Normalize the field
                    mean_val = c['output']['mean'][field_name]
                    std_val = c['output']['std'][field_name]
                    field_norm = (field_data - mean_val) / std_val
                    output_data.append(field_norm[:, None])
                
                if output_data:
                    node_kwargs['output'] = torch.tensor(np.hstack(output_data), dtype=torch.float)
            
            # Add conditional inputs (latent + inlet velocity)
            inlet_vel = np.mean(data[:, self.inlet_vel_indices], axis=0)  # Average inlet velocity
            cond = np.concatenate([self.geom_latents[i], inlet_vel])
            
            # Normalize conditional inputs
            cm, cs = c['cond']['mean'], c['cond']['std']
            cond_n = (cond - cm) / cs
            node_kwargs['cond'] = torch.tensor(cond_n, dtype=torch.float).unsqueeze(0)
            
            processed_data_list.append(Data(**node_kwargs))
        
        return processed_data_list

    def split_val(self, val_size, seed=None):
        """Split the dataset into training and validation sets."""
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
            coef_norm=self.coef_norm,
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

