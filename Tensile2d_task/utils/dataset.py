import numpy as np
import torch
from torch_geometric.data import Dataset, Data

class Tensil_2D_flowDataset(Dataset):
    """
    PyG Dataset for VKI flow SDF data loaded via pyoche MlDataset.

    Modes:
      - 'train': include inputs, pos, outputs, cond, output_scalars; computes normals
      - 'val':   include inputs, pos, outputs, cond, output_scalars; uses provided coef_norm
      - 'test':  include inputs, pos, cond only; uses provided coef_norm

    Each sample keys:
      - 'mesh/points': (2, N)
      - 'mesh/sdf':    (1, N)
      - 'mesh/mach':   (1, N)
      - 'mesh/nut':    (1, N)
      - scalars/... fields

    Normalizations:
      inputs:   [x,y] & sdf -> min-max to [-1,1]
      outputs:  [mach,nut] -> standard
      cond:     geom_latents+[angle_in,mach_out] -> standard
      scalars:  [Pr,Q,Tr,angle_out,eth_is,power] -> standard
    """
    def __init__(
        self,
        Tensil_2d,
        geom_latents,
        mode='train',
        coef_norm=None,
        num_points=None,
        idx_list=None,
        transform=None,
        pre_transform=None
    ):
        super().__init__(None, transform, pre_transform)
        self.Tensil_2d = Tensil_2d
        self._raw_geom_latents = np.array(geom_latents)
        total = len(Tensil_2d)
        self.idx_list = list(range(total)) if idx_list is None else list(idx_list)
        self.geom_latents = self._raw_geom_latents[self.idx_list]
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
        pos_all, sdf_all = [], []
        U1_all, U2_all, sig11_all, sig22_all, sig12_all = [], [], [], [], []
        cond_all, scal_all = [], []
        scalar_keys = ['max_von_mises', 'max_U2_top', 'max_sig22_top']
        for idx in range(len(self.Tensil_2d)):

            item = list(self.Tensil_2d.values())[idx]
            pts = np.array(item.pos)
            sdf = np.array(item.x[:,2])
            pos_all.append(pts)
            sdf_all.append(sdf)
            if self.mode in ('train','val'):

                U1_all.append(np.array(item['U1']).flatten())
                U2_all.append(np.array(item['U2']).flatten())

                sig11_all.append(np.array(item['sig11']).flatten())
                sig22_all.append(np.array(item['sig22']).flatten())
                sig12_all.append(np.array(item['sig12']).flatten())


            angles = [float(item['p']), float(item['p1']), float(item['p2']), float(item['p3']), float(item['p4']), float(item['p5'])]
            cond_all.append(np.concatenate([self.geom_latents[self.idx_list.index(idx)], angles]))
            scal_all.append([float(item[k]) for k in scalar_keys])
        pos_all = np.vstack(pos_all)
        sdf_all = np.hstack(sdf_all)
        pos_min = pos_all.min(axis=0)
        pos_range = pos_all.max(axis=0) - pos_min
        pos_range[pos_range==0] = 1.0
        sdf_min = sdf_all.min()
        sdf_range = sdf_all.max() - sdf_min or 1.0
        out_mean = out_std = None
        if self.mode in ('train','val'):


            U1_all = np.hstack(U1_all)
            U2_all = np.hstack(U2_all)

            sig11_all = np.hstack(sig11_all)
            sig22_all = np.hstack(sig22_all)
            sig12_all = np.hstack(sig12_all)



            out_mean = np.array([U1_all.mean(), U2_all.mean(), sig11_all.mean(), sig22_all.mean(), sig12_all.mean()])
            out_std  = np.array([U1_all.std(), U2_all.std(), sig11_all.std(), sig22_all.std(), sig12_all.std()])
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
            'sdf':    {'min': sdf_min, 'range': sdf_range},
            'output': {'mean': out_mean, 'std': out_std},
            'cond':   {'mean': cond_mean,'std': cond_std},
            'scalars':{'mean': scal_mean,'std': scal_std}
        }

    def process_dataset(self):
        data_list = []
        c = self.coef_norm
        skeys =['max_von_mises', 'max_U2_top', 'max_sig22_top']
        for idx in range(len(self.Tensil_2d)):

            item = list(self.Tensil_2d.values())[idx]
            pts = np.array(item.pos)
            sdf = np.array(item.x[:,2])
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
                'pos':   torch.tensor(pts_n, dtype=torch.float)  # for batching
            }
            if self.mode in ('train','val'):



                U1 = np.array(item['U1']).flatten()
                U2 = np.array(item['U2']).flatten()
                sig11 = np.array(item['sig11']).flatten()
                sig22 = np.array(item['sig22']).flatten()
                sig12 = np.array(item['sig12']).flatten()




                om, os = c['output']['mean'], c['output']['std']
                out_np = np.stack([(U1-om[0])/os[0], (U2-om[1])/os[1], (sig11-om[2])/os[2], (sig22-om[3])/os[3], (sig12-om[4])/os[4]],1)
                node_kwargs['output'] = torch.tensor(out_np, dtype=torch.float)
            angles = [float(item['p']), float(item['p1']), float(item['p2']), float(item['p3']), float(item['p4']), float(item['p5'])]
            raw_cond = np.concatenate([self.geom_latents[idx], angles])
            cm, cs = c['cond']['mean'], c['cond']['std']
            cond_n = (raw_cond - cm)/cs
            node_kwargs['cond'] = torch.tensor(cond_n, dtype=torch.float).unsqueeze(0)
            if self.mode in ('train','val'):
                raw_s = np.array([float(item[k]) for k in skeys])
                sm, ss = c['scalars']['mean'], c['scalars']['std']
                scal_n = (raw_s - sm)/ss
                node_kwargs['output_scalars'] = torch.tensor(scal_n, dtype=torch.float).unsqueeze(0)
            data_list.append(Data(**node_kwargs))
        return data_list

    def split_val(self, val_size, seed=None):
        rng = np.random.RandomState(seed)
        chosen = rng.choice(self.idx_list, size=val_size, replace=False).tolist()
        train_idx = [i for i in self.idx_list if i not in chosen]
        train_ds = Tensil_2D_flowDataset(self.Tensil_2d, self._raw_geom_latents,
                                  mode='train', num_points=self.num_points,
                                  idx_list=train_idx)
        val_ds   = Tensil_2D_flowDataset(self.Tensil_2d, self._raw_geom_latents,
                                  mode='val', coef_norm=self.coef_norm,
                                  num_points=self.num_points,
                                  idx_list=chosen)
        return train_ds, val_ds

    def len(self): return len(self.data_list)

    def get(self, idx): return self.data_list[idx]




class Tensil_2D_SDF(Dataset):
    """
    PyG Dataset for VKI SDF data loaded via pyoche MlDataset.

    Each sample in ml_dataset should support indexing and provide:
      - item['mesh/points']: numpy array of shape (M, 3?) or (M, 2) for coordinates
      - item['mesh/sdf']: numpy array of shape (M,) of SDF values

    Args:
        ml_dataset: pyoche.MlDataset instance (or similar) with ['mesh/points'], ['mesh/sdf']
        is_train: whether to compute normalization from this dataset
        coef_norm: precomputed normalization dict (required if is_train=False)
        num_points: subsample each sample to this number of points
        transform, pre_transform: for PyG compatibility
    """
    def __init__(self, Tensil_2d, is_train=True, coef_norm=None, num_points=None,
                 transform=None, pre_transform=None):
        super(Tensil_2D_SDF, self).__init__(None, transform, pre_transform)
        self.Tensil_2d = Tensil_2d
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
        for item in self.Tensil_2d:


            pts = np.array(item.pos)
            sdf = np.array(item.sdf_continue).reshape(-1,1)


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

        for item in self.Tensil_2d:

            pts = np.array(item.pos)
            sdf = np.array(item.sdf_continue).reshape(-1,1)

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

