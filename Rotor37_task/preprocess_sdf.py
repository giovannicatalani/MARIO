import pyoche as pch
import numpy as np
import torch
import copy
import os
from plaid.containers.dataset import Dataset
from plaid.problem_definition import ProblemDefinition


def triangulate_quad_faces(quad_faces):
    """Convert quadrilateral faces to triangular faces"""
    tri_faces = []
    for quad in quad_faces:
        # Split quad into two triangles: (v0,v1,v2) and (v0,v2,v3)
        tri_faces.append([quad[0], quad[1], quad[2]])
        tri_faces.append([quad[0], quad[2], quad[3]])
    return np.array(tri_faces)

def sample_quad_mesh_random(vertices, quad_faces, n_samples):
    """Sample random points on quadrilateral mesh surface using point cloud utils"""
    import point_cloud_utils as pcu
    
    # Convert quads to triangles for sampling
    tri_faces = triangulate_quad_faces(quad_faces)
    
    # Sample points on triangular mesh using pcu
    fid_surf, bc_surf = pcu.sample_mesh_random(vertices, tri_faces, n_samples)
    points = pcu.interpolate_barycentric_coords(tri_faces, fid_surf, bc_surf, vertices)
    
    return points

def compute_sdf_quad_mesh(points, vertices, quad_faces):
    """Compute signed distance function for quadrilateral mesh using point cloud utils"""
    import point_cloud_utils as pcu
    
    # Convert to triangular mesh for SDF computation
    tri_faces = triangulate_quad_faces(quad_faces)
    
    # Compute signed distances using pcu
    signed_distances, _, _ = pcu.signed_distance_to_mesh(points, vertices, tri_faces)
    
    return signed_distances

def get_global_bounding_box(dataset, ids):
    """Get global bounding box across all meshes with some padding"""
    global_min = np.array([float('inf')] * 3)
    global_max = np.array([float('-inf')] * 3)
    
    for idx in ids:
        sample = dataset[idx]
        nodes = sample.get_nodes()
        
        current_min = nodes.min(axis=0)
        current_max = nodes.max(axis=0)
        
        global_min = np.minimum(global_min, current_min)
        global_max = np.maximum(global_max, current_max)
    
    # Add padding (20% of range on each side)
    range_size = global_max - global_min
    padding = 0.2 * range_size
    
    global_min -= padding
    global_max += padding
    
    return global_min, global_max

def get_sdf_quad(sample_idx, dataset, global_min, global_max, output_dir):
    """
    Compute SDF for quadrilateral mesh and save results
    """
    print(f"Processing sample {sample_idx}")
    
    sample = dataset[sample_idx]
    nodes = sample.get_nodes()  # vertices
    elements = sample.get_elements()['QUAD_4']  # quadrilateral faces
    
    # Convert to numpy if needed
    if torch.is_tensor(nodes):
        vm = nodes.numpy().astype(np.float64)
    else:
        vm = nodes.astype(np.float64)
        
    if torch.is_tensor(elements):
        fm = elements.numpy().astype(np.int32)
    else:
        fm = elements.astype(np.int32)
    
    # Parameters for sampling
    num_vol_pts = int(vm.shape[0])
    num_surf_pts = int(vm.shape[0] * 3)
    
    print(f"Original mesh bounds: min={vm.min(axis=0)}, max={vm.max(axis=0)}")
    
    # Normalize to global bounding box
    normalized_vm = copy.deepcopy(vm)
    for i in range(3):
        normalized_vm[:, i] = (vm[:, i] - global_min[i]) / (global_max[i] - global_min[i])
        normalized_vm[:, i] = normalized_vm[:, i] * 1.2 - 0.6  # Scale to [-0.6, 0.6]
    print(f"Normalized mesh bounds: min={normalized_vm.min(axis=0)}, max={normalized_vm.max(axis=0)}")
    
    # Sample volume points uniformly
    batch = min(num_vol_pts, int(2**17))
    points = np.random.uniform(-0.8, 0.8, size=(batch, 3))
    sdf = compute_sdf_quad_mesh(points, normalized_vm, fm)
    
    while sdf.shape[0] < num_vol_pts:
        p1 = np.random.uniform(-0.8, 0.8, size=(batch, 3))
        sdf1 = compute_sdf_quad_mesh(p1, normalized_vm, fm)
        sdf = np.concatenate((sdf, sdf1), axis=0)
        points = np.concatenate((points, p1), axis=0)
    
    points = points[:num_vol_pts]
    sdf = sdf[:num_vol_pts]
    
    # Sample surface points with different variances
    batch = min(num_surf_pts, int(2**17))
    variances = [0.005, 0.0005]
    
    for var in variances:
        while sdf.shape[0] < (num_vol_pts + num_surf_pts):
            # Sample points on surface
            p2 = sample_quad_mesh_random(normalized_vm, fm, batch)
            
            # Add noise
            p2 += np.random.normal(scale=np.sqrt(var), size=(p2.shape[0], 3))
            
            # Compute SDF
            sdf2 = compute_sdf_quad_mesh(p2, normalized_vm, fm)
            
            # Balance positive and negative samples
            p2_neg = p2[sdf2 < 0][:batch//2, :]
            p2_pos = p2[sdf2 > 0][:batch//2, :]
            sdf2_neg = sdf2[sdf2 < 0][:batch//2]
            sdf2_pos = sdf2[sdf2 > 0][:batch//2]
            
            sdf = np.concatenate((sdf, sdf2_neg, sdf2_pos), axis=0)
            points = np.concatenate((points, p2_neg, p2_pos), axis=0)
    
    # Filter points with reasonable SDF values
    valid_mask = sdf <= 1
    points = points[valid_mask]
    sdf = sdf[valid_mask]
    
    # Truncate to desired number of points
    total_points = num_vol_pts + num_surf_pts
    points = points[:total_points]
    sdf = sdf[:total_points]
    
    # Statistics
    neg_count = np.sum(sdf < 0)
    pos_count = np.sum(sdf > 0)
    
    print(f"Points range: min={points.min()}, max={points.max()}")
    print(f"SDF range: min={sdf.min()}, max={sdf.max()}")
    print(f"Negative samples: {neg_count/(neg_count+pos_count)*100:.2f}%")
    print(f"Total points: {points.shape[0]}")
    
    return points, sdf



if __name__ == "__main__":
    
    dataset = Dataset()
    dataset._load_from_dir_('/scratch/dmsm/gi.catalani/Rotor37/dataset', verbose = True)

    problem = ProblemDefinition()
    problem._load_from_dir_('/scratch/dmsm/gi.catalani/Rotor37/problem_definition')

    ids_train = problem.get_split('train_1000')
    ids_test  = problem.get_split('test')


    # Combine all IDs for global bounding box computation
    all_ids = ids_train + ids_test

    print("Computing global bounding box...")
    global_min, global_max = get_global_bounding_box(dataset, all_ids)
    print(f"Global bounding box: min={global_min}, max={global_max}")

    # Create output directory
    output_dir = "/scratch/dmsm/gi.catalani/Rotor37/pyoche_data"
    os.makedirs(output_dir, exist_ok=True)

    # Process training samples
    print("Processing training samples...")
    for i, sample_idx in enumerate(ids_train):
        
        points, sdf = get_sdf_quad(sample_idx, dataset, global_min, global_max, output_dir)
        # Create sample dictionary with coordinates
        s = pch.Sample.from_dict({
            'point_cloud_field/coordinates': points,
        })
        
        # Add SDF to point_cloud_field
        s['point_cloud_field/sdf'] = sdf[np.newaxis]
        
        # Save the sample
        output_dir = '/scratch/dmsm/gi.catalani/Rotor37/pyoche_data/Rotor37_sdf_train.pch'
        os.makedirs(output_dir, exist_ok=True)
        s.save_h5file(f'{output_dir}/_sample_{sample_idx:03d}.h5')
    # Process test samples
    print("Processing test samples...")
    for i, sample_idx in enumerate(ids_test):
        
        points, sdf = get_sdf_quad(sample_idx, dataset, global_min, global_max, output_dir)
        # Create sample dictionary with coordinates
        s = pch.Sample.from_dict({
            'point_cloud_field/coordinates': points,
        })
        
        # Add SDF to point_cloud_field
        s['point_cloud_field/sdf'] = sdf[np.newaxis]
        
        # Save the sample
        output_dir = '/scratch/dmsm/gi.catalani/Rotor37/pyoche_data/Rotor37_sdf_test.pch'
        os.makedirs(output_dir, exist_ok=True)
        s.save_h5file(f'{output_dir}/_sample_{sample_idx:03d}.h5')

    print("SDF computation completed!")