import numpy as np
import torch
from torch_geometric.data import Dataset, Data

class RotorFlowDataset(Dataset):
    """
    
    """
    def __init__(
        self,
        ml_dataset,
        idxs,
        geom_latents,
        mode='train',
        coef_norm=None,
        num_points=None,
        transform=None,
        pre_transform=None
    ):
        super().__init__(None, transform, pre_transform)
        self.ml_dataset = ml_dataset
        self._raw_geom_latents = np.array(geom_latents)
        self.idxs= idxs
        self.geom_latents = self._raw_geom_latents
        self.mode = mode
        self.num_points = num_points
        # norms
        if mode == 'train':
            self.coef_norm = self.compute_norm_params()
        else:
            if coef_norm is None:
                raise ValueError("coef_norm required for non-train modes")
            self.coef_norm = coef_norm
        self.data_list = self.process_dataset()

    def compute_norm_params(self):
        pos_all, normals_all = [], []
        rho_all, pressure_all, temperature_all = [], [], []
        cond_all, scal_all = [], []
        scalar_keys = ['Massflow', 'Compression_ratio', 'Efficiency']
        for i,idx in enumerate(self.idxs):
            item = self.ml_dataset[idx]
            pts = np.array(item.get_nodes())
            normals_x = np.array(item.get_field('NormalsX')).T.flatten()
            normals_y = np.array(item.get_field('NormalsY')).T.flatten()
            normals_z = np.array(item.get_field('NormalsZ')).T.flatten()
            normals = np.stack([normals_x, normals_y, normals_z], axis=1)
            pos_all.append(pts)
            normals_all.append(normals)
            if self.mode in ('train','val'):
                rho_all.append(np.array(item.get_field('Density')).T.flatten())
                pressure_all.append(np.array(item.get_field('Pressure')).T.flatten())
                temperature_all.append(np.array(item.get_field('Temperature')).T.flatten())
                
            cond_in = [float(item.get_scalar('Omega')), float(item.get_scalar('P'))]
            cond_all.append(np.concatenate([self.geom_latents[i], cond_in]))
            scal_all.append([float(item.get_scalar(k)) for k in scalar_keys])
        pos_all = np.vstack(pos_all)
        normals_all = np.vstack(normals_all)
        pos_min = pos_all.min(axis=0)
        pos_range = pos_all.max(axis=0) - pos_min
        pos_range[pos_range==0] = 1.0
        normals_min = normals_all.min(axis=0)
        normals_max = normals_all.max(axis=0)
        normals_range = normals_max - normals_min
        out_mean = out_std = None
        if self.mode in ('train','val'):
            rho_all = np.hstack(rho_all)
            pressure_all = np.hstack(pressure_all)
            temperature_all = np.hstack(temperature_all)
            out_mean = np.array([rho_all.mean(), pressure_all.mean(), temperature_all.mean()])
            out_std  = np.array([rho_all.std(), pressure_all.std(), temperature_all.std()])
            out_std[out_std==0] = 1.0
        cond_all = np.vstack(cond_all)
        cond_mean = cond_all.mean(axis=0)
        cond_std  = cond_all.std(axis=0)
        cond_std[cond_std==0] = 1.0
        scal_all = np.vstack(scal_all)
        scal_mean = scal_all.mean(axis=0)
        scal_std  = scal_all.std(axis=0)
        scal_std[scal_std==0] = 1.0
        return {
            'pos':    {'min': pos_min, 'range': pos_range},
            'normals': {'min': normals_min, 'range': normals_range},
            'output': {'mean': out_mean, 'std': out_std},
            'cond':   {'mean': cond_mean,'std': cond_std},
            'scalars':{'mean': scal_mean,'std': scal_std}
        }

    def process_dataset(self):
        data_list = []
        c = self.coef_norm
        skeys = ['Massflow', 'Compression_ratio', 'Efficiency']
        for i,idx in enumerate(self.idxs):
            item = self.ml_dataset[idx]
            pts = np.array(item.get_nodes())
            normals_x = np.array(item.get_field('NormalsX')).T.flatten()
            normals_y = np.array(item.get_field('NormalsY')).T.flatten()
            normals_z = np.array(item.get_field('NormalsZ')).T.flatten()
            normals = np.stack([normals_x, normals_y, normals_z], axis=1)
            
            if self.num_points and pts.shape[0] > self.num_points:
                sel = np.random.choice(pts.shape[0], self.num_points, replace=False)
                pts, normals = pts[sel], normals[sel]
            pmin, pr = c['pos']['min'], c['pos']['range']
            nmin, nr = c['normals']['min'], c['normals']['range']
            pts_n = 2*(pts - pmin)/pr - 1
            normals_n = 2*((normals - nmin)/nr) - 1
            # per-node features
            input_feats = torch.tensor(np.concatenate([pts_n, normals_n],axis=1), dtype=torch.float)
            node_kwargs = {
                'input': input_feats,
                'pos':   torch.tensor(pts_n, dtype=torch.float)  # for batching
            }
            if self.mode in ('train','val'):
                rho = np.array(item.get_field('Density')).T.flatten()
                pressure = np.array(item.get_field('Pressure')).T.flatten()
                temperature = np.array(item.get_field('Temperature')).T.flatten()
                
                om, os = c['output']['mean'], c['output']['std']
                out_np = np.stack([(rho-om[0])/os[0], (pressure-om[1])/os[1], (temperature-om[2])/os[2]],1)
                node_kwargs['output'] = torch.tensor(out_np, dtype=torch.float)
            cond_in = [float(item.get_scalar('Omega')), float(item.get_scalar('P'))]
            raw_cond = np.concatenate([self.geom_latents[i], cond_in])
            cm, cs = c['cond']['mean'], c['cond']['std']
            cond_n = (raw_cond - cm)/cs
            node_kwargs['cond'] = torch.tensor(cond_n, dtype=torch.float).unsqueeze(0)
            if self.mode in ('train','val'):
                raw_s = np.array([float(item.get_scalar(k)) for k in skeys])
                sm, ss = c['scalars']['mean'], c['scalars']['std']
                scal_n = (raw_s - sm)/ss
                node_kwargs['output_scalars'] = torch.tensor(scal_n, dtype=torch.float).unsqueeze(0)
            data_list.append(Data(**node_kwargs))
        return data_list

    def len(self): return len(self.data_list)

    def get(self, idx): return self.data_list[idx]
    
    
class RotorSDFDataset(Dataset):
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
        super(RotorSDFDataset, self).__init__(None, transform, pre_transform)
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
            pts = np.array(item['point_cloud_field/coordinates'])
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
            pts = np.array(item['point_cloud_field/coordinates'])
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

