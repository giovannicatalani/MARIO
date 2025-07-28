import numpy as np
import torch
from torch_geometric.data import Dataset, Data


class profileSDFDataset(Dataset):
    """
    PyG Dataset for 2D PROFILE data loaded via pyoche MlDataset.

    Each sample in ml_dataset should support indexing and provide:
      - item['point_cloud_field/coordinates']: numpy array of shape (M, 3?) or (M, 2) for coordinates
      - item['point_cloud_field/sdf']: numpy array of shape (M,) of SDF values

    Args:
        ml_dataset: pyoche.MlDataset instance (or similar) with ['mesh/points'], ['mesh/sdf']
        is_train: whether to compute normalization from this dataset
        coef_norm: precomputed normalization dict (required if is_train=False)
        num_points: subsample each sample to this number of points
        transform, pre_transform: for PyG compatibility
    """
    def __init__(self, ml_dataset, is_train=True, coef_norm=None, num_points=None,
                 transform=None, pre_transform=None):
        super(profileSDFDataset, self).__init__(None, transform, pre_transform)
        self.ml_dataset = ml_dataset
        self.num_points = num_points
        self.is_train = is_train

        if self.is_train:
            self.coef_norm = self.compute_norm_params()
        else:
            if coef_norm is None:
                raise ValueError("coef_norm must be provided when is_train is False.")
            self.coef_norm = coef_norm

        self.data_list = self.process_dataset()

    def compute_norm_params(self):
        """Compute normalization parameters (pos min/max, sdf mean/std) from all samples."""
        all_positions = []
        all_sdf = []
        for idx in range(len(self.ml_dataset)):
            item = self.ml_dataset[idx]
            pts = np.array(item['point_cloud_field/coordinates']).T
            sdf = np.array(item['point_cloud_field/sdf']).T
            all_positions.append(pts)
            all_sdf.append(sdf)
        all_positions = np.vstack(all_positions)
        all_sdf = np.vstack(all_sdf)

        pos_min = all_positions.min(axis=0)
        pos_max = all_positions.max(axis=0)
        sdf_mean = all_sdf.mean(axis=0)
        sdf_std = all_sdf.std(axis=0)

        # Avoid zero division
        pos_range = pos_max - pos_min
        pos_range[pos_range == 0] = 1.0
        if sdf_std == 0:
            sdf_std = 1.0

        coef_norm = {
            'pos_norm': {
                'min': pos_min,
                'max': pos_max
            },
            'mean': sdf_mean,
            'std': sdf_std
        }

        print("Normalization parameters computed:")
        print(f"Position min: {pos_min}, max: {pos_max}")
        print(f"SDF mean: {sdf_mean}, std: {sdf_std}")
        return coef_norm

    def process_dataset(self):
        """Normalize, filter, subsample and convert each sample to a PyG Data object."""
        data_list = []
        pos_min = self.coef_norm['pos_norm']['min']
        pos_max = self.coef_norm['pos_norm']['max']
        sdf_mean = self.coef_norm['mean']
        sdf_std = self.coef_norm['std']

        # Handle zero-range safeguards
        pos_range = pos_max - pos_min
        pos_range[pos_range == 0] = 1.0

        for idx in range(len(self.ml_dataset)):
            item = self.ml_dataset[idx]
            pts = np.array(item['point_cloud_field/coordinates']).T
            sdf = np.array(item['point_cloud_field/sdf']).T

            # Normalize positions to [-1, 1]
            pts_norm = 2 * (pts - pos_min) / pos_range - 1
            # Normalize SDF to zero mean, unit std
            sdf_norm = (sdf - sdf_mean) / sdf_std

        
            # Subsample if requested
            if self.num_points is not None and pts_norm.shape[0] > self.num_points:
                indices = np.random.choice(pts_norm.shape[0], self.num_points, replace=False)
                pts_norm = pts_norm[indices]
                sdf_norm = sdf_norm[indices]

            data = Data(
                pos=torch.tensor(pts_norm, dtype=torch.float),
                input=torch.tensor(pts_norm, dtype=torch.float),
                output=torch.tensor(sdf_norm, dtype=torch.float),
                is_airfoil=torch.zeros((pts_norm.shape[0],), dtype=torch.float)
            )
            data_list.append(data)
        return data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
    
    

class profileFlowDataset(Dataset):
    """
    PyG Dataset for VKI flow SDF data loaded via pyoche MlDataset.

    Modes:
      - 'train': include inputs, pos, outputs, cond; computes normals
      - 'val':   include inputs, pos, outputs, cond; uses provided coef_norm
      - 'test':  include inputs, pos, cond only; uses provided coef_norm

    Each sample keys:
      - 'point_cloud_field/coordinates': (2, N)
      - 'point_cloud_field/sdf':    (1, N)
      - Output fields: 'Mach', 'Pressure', 'Velocity-x', 'Velocity-y'

    Normalizations:
      inputs:   [x,y] & sdf -> min-max to [-1,1]
      outputs:  [Mach, Pressure, Velocity-x, Velocity-y] -> standard
      cond:     geom_latents -> standard
    """
    def __init__(
        self,
        ml_dataset,
        geom_latents,
        mode='train',
        coef_norm=None,
        num_points=None,
        idx_list=None,
        transform=None,
        pre_transform=None
    ):
        super().__init__(None, transform, pre_transform)
        self.ml_dataset = ml_dataset
        self._raw_geom_latents = np.array(geom_latents)
        total = len(ml_dataset)
        self.idx_list = list(range(total)) if idx_list is None else list(idx_list)
        self.geom_latents = self._raw_geom_latents[self.idx_list]
        self.mode = mode
        self.num_points = num_points
        # Output field names
        self.output_fields = ['Mach', 'Pressure', 'Velocity-x', 'Velocity-y']
        
        # norms
        if mode == 'train':
            self.coef_norm = self.compute_norm_params()
        else:
            if coef_norm is None:
                raise ValueError("coef_norm required for non-train modes")
            self.coef_norm = coef_norm
        self.data_list = self.process_dataset()

    def compute_norm_params(self):
        pos_all, sdf_all = [], []
        output_fields_all = {field: [] for field in self.output_fields}
        cond_all = []
        
        for idx in self.idx_list:
            item = self.ml_dataset[idx]
            pts = np.array(item['point_cloud_field/coordinates']).T
            sdf = np.array(item['point_cloud_field/sdf']).T.flatten()
            pos_all.append(pts)
            sdf_all.append(sdf)
            
            if self.mode in ('train', 'val'):
                for field in self.output_fields:
                    try:
                        field_data = np.array(item[f'point_cloud_field/{field}']).T.flatten()
                        output_fields_all[field].append(field_data)
                    except (KeyError, TypeError) as e:
                        print(f"Warning: Could not get field {field}: {e}")
            
            # Add latent conditions without scalars
            cond_all.append(self.geom_latents[self.idx_list.index(idx)])
        
        pos_all = np.vstack(pos_all)
        sdf_all = np.hstack(sdf_all)
        pos_min = pos_all.min(axis=0)
        pos_range = pos_all.max(axis=0) - pos_min
        pos_range[pos_range==0] = 1.0
        sdf_min = sdf_all.min()
        sdf_range = sdf_all.max() - sdf_min or 1.0
        
        # Process output fields
        output_means = {}
        output_stds = {}
        
        if self.mode in ('train', 'val'):
            for field in self.output_fields:
                if output_fields_all[field]:
                    field_data = np.hstack(output_fields_all[field])
                    output_means[field] = field_data.mean()
                    output_stds[field] = field_data.std()
                    if output_stds[field] == 0:
                        output_stds[field] = 1.0
                else:
                    output_means[field] = 0.0
                    output_stds[field] = 1.0
        
        cond_all = np.vstack(cond_all)
        cond_mean = cond_all.mean(axis=0)
        cond_std = cond_all.std(axis=0)
        cond_std[cond_std==0] = 1.0
        
        return {
            'pos': {'min': pos_min, 'range': pos_range},
            'sdf': {'min': sdf_min, 'range': sdf_range},
            'output': {'mean': output_means, 'std': output_stds},
            'cond': {'mean': cond_mean, 'std': cond_std}
        }

    def process_dataset(self):
        data_list = []
        c = self.coef_norm
        
        for i, idx in enumerate(self.idx_list):
            item = self.ml_dataset[idx]
            pts = np.array(item['point_cloud_field/coordinates']).T
            sdf = np.array(item['point_cloud_field/sdf']).T.flatten()
            
            if self.num_points and pts.shape[0] > self.num_points:
                sel = np.random.choice(pts.shape[0], self.num_points, replace=False)
                pts, sdf = pts[sel], sdf[sel]
            
            pmin, pr = c['pos']['min'], c['pos']['range']
            smin, sr = c['sdf']['min'], c['sdf']['range']
            pts_n = 2*(pts - pmin)/pr - 1
            sdf_n = 2*((sdf - smin)/sr) - 1
            
            # per-node features
            input_feats = torch.tensor(np.concatenate([pts_n, sdf_n[:,None]],1), dtype=torch.float)
            node_kwargs = {
                'input': input_feats,
                'pos': torch.tensor(pts_n, dtype=torch.float)  # for batching
            }
            
            if self.mode in ('train', 'val'):
                output_data = []
                for field in self.output_fields:
                    try:
                        field_data = np.array(item[f'point_cloud_field/{field}']).T.flatten()
                        if self.num_points and field_data.shape[0] > self.num_points:
                            field_data = field_data[sel]
                        
                        # Normalize the field
                        mean_val = c['output']['mean'][field]
                        std_val = c['output']['std'][field]
                        field_norm = (field_data - mean_val) / std_val
                        output_data.append(field_norm[:, None])
                    except (KeyError, TypeError):
                        # Fill with zeros if field is missing
                        output_data.append(np.zeros((pts_n.shape[0], 1)))
                
                if output_data:
                    node_kwargs['output'] = torch.tensor(np.hstack(output_data), dtype=torch.float)
            
            # Add latent conditions
            cm, cs = c['cond']['mean'], c['cond']['std']
            cond_n = (self.geom_latents[i] - cm) / cs
            node_kwargs['cond'] = torch.tensor(cond_n, dtype=torch.float).unsqueeze(0)
            
            data_list.append(Data(**node_kwargs))
        
        return data_list

    def split_val(self, val_size, seed=None):
        rng = np.random.RandomState(seed)
        chosen = rng.choice(self.idx_list, size=val_size, replace=False).tolist()
        train_idx = [i for i in self.idx_list if i not in chosen]
        train_ds = profileFlowDataset(self.ml_dataset, self._raw_geom_latents,
                                  mode='train', num_points=self.num_points,
                                  idx_list=train_idx)
        val_ds = profileFlowDataset(self.ml_dataset, self._raw_geom_latents,
                                mode='val', coef_norm=self.coef_norm,
                                num_points=self.num_points,
                                idx_list=chosen)
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

