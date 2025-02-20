from copy import deepcopy
import pyoche as pch
from tqdm import tqdm
import numpy as np
from scipy.spatial import cKDTree
from pyoche.ml.normalize import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
from typing import Literal

def make_bl_layer(sample, thickness, decay: Literal['linear', 'exponential', 'squared']='squared'):
    if not (thickness > 0 and thickness <= 1):
        raise ValueError('Thickness should be in ]0;1]')

    def threshold(var, value):
        mask = var>value
        var[~mask] = 0
        var[mask] = (var[mask]-value)/(var[mask].max()-value)
        return var
    df = (
        sample['point_cloud_field/distance_function'].max()
        -sample['point_cloud_field/distance_function']
    )
    df = df/df.max()
    if decay == 'squared':
        return threshold(df, 1-thickness)**2
    elif decay == 'linear':
        return threshold(df, 1-thickness)
    elif decay == 'exponential':
        return (np.exp(threshold(df, 1-thickness))-1)/(np.exp(1)-1)
    else:
        return ValueError('Decay types are: squared, linear or exponential')

def assign_normals(sample, sub_array_indices, fade=True, flow_relative=False):
    surf = sample['point_cloud_field/is_surf'].astype(bool)[0]
    surface_positions = sample['point_cloud_field/coordinates'][:, surf].T
    domain_positions = sample['point_cloud_field/coordinates'][:, sub_array_indices].T
    

    # Step 2: Build KDTree and query closest points
    tree = cKDTree(surface_positions)
    _, closest_surface_indices = tree.query(domain_positions, k=2)
   
    # Step 3: Calculate closest points
    A = surface_positions[closest_surface_indices[:, 0]]
    B = surface_positions[closest_surface_indices[:, 1]]
    AB = B - A
    AP = domain_positions - A
    t = np.einsum('ij,ij->i', AP, AB) / np.einsum('ij,ij->i', AB, AB)
    t = np.clip(t, 0, 1)
    closest_points = A + (t[:, np.newaxis] * AB)
    

    # Step 4: Calculate vectors and normali
    vectors_to_closest = closest_points - domain_positions
    norms = np.linalg.norm(vectors_to_closest, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    unit_vectors_to_closest = vectors_to_closest / norms
    
    # Step 5: Fade factor calculation (optional)
    if fade:
        tmp_scaler = MinMaxScaler()
        tmp_sdf = tmp_scaler.fit_transform(
            sample['point_cloud_field/distance_function'][:, sub_array_indices].T
        )[:, 0]
        
        factor = (tmp_sdf.max() - tmp_sdf)
    else:
        factor = 1
    
    sample['point_cloud_field/normals'][0, sub_array_indices] = unit_vectors_to_closest[:, 0] * factor
    sample['point_cloud_field/normals'][1, sub_array_indices] = unit_vectors_to_closest[:, 1] * factor

    if flow_relative:
        alpha = np.arctan2(*sample['point_cloud_field/inlet_velocity'][:, 0])
        tmpx, tmpy = sample['point_cloud_field/normals']

        sample['point_cloud_field/normals'][0] = tmpx * np.cos(alpha) + tmpy * np.sin(alpha)
        sample['point_cloud_field/normals'][1] = tmpy * np.cos(alpha) - tmpx * np.sin(alpha)

    return sample


def compute_airfoil_metrics(sample):
    # Extract position and normals directly from the data object
    idx = np.squeeze(sample["point_cloud_field/is_surf"]) == 1
    nodes = sample["point_cloud_field/coordinates"].T[idx]
    normals = np.vstack((sample['point_cloud_field/normals'][0], sample['point_cloud_field/normals'][1])).T[
        idx
    ]  # Assuming normal vectors are stored here

    # Classify nodes into upper and lower surfaces based on the y-component of normals
    upper_nodes = nodes[normals[:, 1] < 0]
    lower_nodes = nodes[normals[:, 1] > 0]

    max_thickness = 0.0
    max_camber = 0.0

    # Thickness and camber at specific chord fractions
    chord_positions = [0.05, 0.1, 0.15, 0.25, 0.4, 0.5, 0.7, 0.8, 0.9]
    thickness_at_positions = []
    camber_at_positions = []

    if len(upper_nodes) > 0 and len(lower_nodes) > 0:
        # Sort nodes by x-coordinate
        upper_nodes = upper_nodes[np.argsort(upper_nodes[:, 0])]
        lower_nodes = lower_nodes[np.argsort(lower_nodes[:, 0])]

        # Calculate maximum thickness
        for ux, uy in upper_nodes:
            distances = np.abs(lower_nodes[:, 0] - ux)
            idx = np.argmin(distances)
            lx, ly = lower_nodes[idx]
            thickness = np.abs(uy - ly)
            max_thickness = max(max_thickness, thickness)

        # Interpolation for camber and thickness calculation at specific positions
        common_x = np.linspace(
            upper_nodes[:, 0].min(), upper_nodes[:, 0].max(), num=500
        )
        upper_interp = np.interp(common_x, upper_nodes[:, 0], upper_nodes[:, 1])
        lower_interp = np.interp(common_x, lower_nodes[:, 0], lower_nodes[:, 1])
        mean_camber_line = (upper_interp + lower_interp) / 2
        thickness_distribution = np.abs(upper_interp - lower_interp)
        chord_line = np.linspace(mean_camber_line[0], mean_camber_line[-1], num=500)

        # Calculate maximum camber
        max_camber = np.max(np.abs(mean_camber_line - chord_line))

        
        for position in chord_positions:
            # Find the closest x-coordinate to the chord fraction
            chord_x = position * (upper_nodes[:, 0].max() - upper_nodes[:, 0].min())

            # Interpolate the upper and lower surfaces at the chord fraction position
            upper_y_at_x = np.interp(chord_x, upper_nodes[:, 0], upper_nodes[:, 1])
            lower_y_at_x = np.interp(chord_x, lower_nodes[:, 0], lower_nodes[:, 1])

            # Calculate thickness at the specific position
            thickness_at_x = np.abs(upper_y_at_x - lower_y_at_x)
            thickness_at_positions.append(thickness_at_x)

            # Calculate camber at the specific position (difference from the chord line)
            camber_at_x = np.abs((upper_y_at_x + lower_y_at_x) / 2)
            camber_at_positions.append(camber_at_x)

        return np.concatenate(
            (np.array([max_camber, max_thickness]), camber_at_positions, thickness_at_positions)
        )

def plot_tricontourf(
        coords, field, surface_mask=None, 
        show_plot=True, for_wandb=False, cmap='RdBu'
    ):
    if coords.shape[0] != 2 or field.shape[0] != 1:
        raise ValueError("Coordinates must be of shape (2, n) and field must be of shape (1, n).")

    x, y = coords
    z = field.flatten()
    plt.figure(figsize=(8, 6))
    plt.tricontourf(x, y, z, levels=30, cmap=cmap)
    plt.colorbar(label="Field Value")
    plt.title("Tricontourf Plot of Field Over 2D Point Cloud")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    if surface_mask is not None:
        white = matplotlib.colors.ListedColormap("white")
        plt.tricontourf(x[surface_mask], y[surface_mask], z[surface_mask], levels=20, cmap=white)
        plt.scatter(x[surface_mask], y[surface_mask], c='k', s=1)

    if for_wandb:
        return plt.gcf()


    if show_plot:
        plt.axis("equal")
        plt.show()

    # return plt.gcf(), plt.gca()
    

def train_step_function(self, batch, is_train=True):
    yhat = self.model.forward(batch.x, batch.z)[0]
    

    if is_train:
        if self.loss.requires_model_params:
            self.loss(
                yhat, batch.y, 
                self.model.parameters(),
                n_samples_batch=batch.num_graphs
            )
        else:
            self.loss(
                yhat, batch.y,
                n_samples_batch=batch.num_graphs
            )
        self.loss.main.backward()
    else:
        self.val_loss(
            yhat, batch.y, 
            n_samples_batch=batch.num_graphs
        )
    
    return yhat[batch.batch <= self.cb_batch_size].detach().cpu()


def process_dataset(dataset):
    feature_indices = {
        'coordinates': slice(0, 2),  # x, y coordinates
        'inlet_velocity': slice(2, 4),  # velocity components
        'distance_function': 4,  # distance to airfoil
        'normals': slice(5, 7),  # normal components
        'velocity': slice(7, 9),  # target velocity components
        'pressure': 9,  # target pressure
        'turbulent_viscosity': 10,  # target turbulent viscosity
        'is_surf': 11  # boolean flag for airfoil points
    }

    samples, names = deepcopy(dataset)

    # codes = np.load('/scratch/daep/j.fesquet/git_repos/MARIO/data/modulations/scarce_test_8.npz')['val_modulations']
    # codes = np.load('/scratch/daep/j.fesquet/git_repos/MARIO/data/modulations/scarce_train_8.npz')['train_modulations']

    sample_list = []
    for i, sample in tqdm(enumerate(samples), desc='processing samples'):
        dict_sample = {}
        for key, value in feature_indices.items():
            dict_sample[key] = np.atleast_2d(sample.T[value])

        pyoche_sample = pch.MlSample.from_dict({
            "point_cloud_field": dict_sample,
            # "scalars": {"geometry_code": codes[i]}
        })

        pyoche_sample.name = names[i]
        sample_list.append(pyoche_sample)
        break
    return sample_list




######################################

def train_step_function_mlp(self, batch, is_train=True):
    yhat = self.model.forward(batch.x)[0]

    if is_train:
        if self.loss.requires_model_params:
            self.loss(yhat, batch.y, self.model.parameters())
        else:
            self.loss(yhat, batch.y)
        self.loss.main.backward()
    else:
        self.val_loss(yhat, batch.y)
    
    
    return yhat[batch.batch <= self.cb_batch_size].detach().cpu()