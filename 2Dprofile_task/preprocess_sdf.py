from plaid.containers.dataset import Dataset
from plaid.problem_definition import ProblemDefinition
import os
import numpy as np
from collections import defaultdict

class Polygon:
    def __init__(self, vertices):
        """
        Initialize a polygon.
        
        :param vertices: (N,2) array-like list of vertices.
        """
        self.vertices = np.array(vertices)
    
    def sdf(self, points):
        """
        Compute the signed distance for a set of 2D points with respect to this polygon.
        This is a vectorized implementation of the algorithm.
        
        :param points: (M,2) array of sample points.
        :return: (M,) array of signed distances.
        """
        A = self.vertices
        B = np.roll(self.vertices, -1, axis=0)
        AB = B - A
        AB_norm_sq = np.sum(AB ** 2, axis=1)
        
        diff = points[:, None, :] - A[None, :, :]
        dot = np.sum(diff * AB[None, :, :], axis=2)
        t = np.clip(dot / (AB_norm_sq[None, :] + 1e-12), 0, 1)
        projection = A[None, :, :] + t[:, :, None] * AB[None, :, :]
        distances = np.linalg.norm(points[:, None, :] - projection, axis=2)
        min_distance = np.min(distances, axis=1)
        
        # Inside test via angle-sum method.
        v1 = A[None, :, :] - points[:, None, :]
        v2 = B[None, :, :] - points[:, None, :]
        dot_angles = np.sum(v1 * v2, axis=2)
        cross_angles = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
        angles = np.arctan2(cross_angles, dot_angles)
        angle_sum = np.sum(angles, axis=1)
        inside = np.abs(angle_sum) > 1
        
        # Negative distance inside.
        sdf = np.where(inside, -min_distance, min_distance)
        return sdf

    def sample_points(self, num_points=10000, bbox_extension=(0.5, 0.5)):
        """
        Sample points around the polygon. First, a set of uniformly random points
        is drawn from an extended bounding box. Optionally, points can be sampled near
        the polygon's boundary with some noise added.
        
        :param num_points: Total number of points.
        :param bbox_extension: How much to extend the polygon's bounding box.
        :return: (points, sdf) where points is an (M,2) array and sdf is (M,).
        """
        
        perturb_variances = [0.05, 0.02, 0.005]
        
        min_corner = self.vertices.min(axis=0) - np.array(bbox_extension)
        max_corner = self.vertices.max(axis=0) + np.array(bbox_extension)
        
        # Uniform sampling.
        num_uniform = num_points // 5
        uniform_points = np.random.uniform(min_corner, max_corner, size=(num_uniform, 2))
        uniform_sdf = self.sdf(uniform_points)
        
        # Perturbed sampling around each vertex.
        perturbed_points = []
        for var in perturb_variances:
            num_repeat = num_points // self.vertices.shape[0]
            noise = np.random.normal(scale=np.sqrt(var), size=(self.vertices.shape[0] * num_repeat, 2))
            pts = np.repeat(self.vertices, num_repeat, axis=0) + noise
            perturbed_points.append(pts)
        
        if perturbed_points:
            perturbed_points = np.vstack(perturbed_points)
            perturbed_sdf = self.sdf(perturbed_points)
        else:
            perturbed_points = np.empty((0,2))
            perturbed_sdf = np.empty((0,))
        
        all_points = np.vstack((uniform_points, perturbed_points))
        all_sdf = np.hstack((uniform_sdf, perturbed_sdf))
        return all_points, all_sdf


def compute_airfoil_sdf(sample):
    """
    Compute the signed distance function for a sample using the airfoil boundary nodes.
    
    Args:
        sample: The sample containing nodes, elements, and nodal tags
        
    Returns:
        sdf_values: Array of SDF values for all nodes
    """
    # Extract data from sample
    nodes = sample.get_nodes()
    nodal_tags = sample.get_nodal_tags()
    elements = sample.get_elements()
    
    # Get airfoil node indices
    airfoil_indices = nodal_tags['Airfoil']
    airfoil_nodes = nodes[airfoil_indices]
    
    # Order the airfoil nodes to form a closed polygon
    ordered_airfoil_nodes = order_boundary_nodes(nodes, airfoil_nodes, elements, airfoil_indices)
    
    # Create polygon object
    airfoil_polygon = Polygon(ordered_airfoil_nodes)
    
    # Compute SDF for all nodes
    sdf_values = airfoil_polygon.sdf(nodes)
    
    return sdf_values

def order_boundary_nodes(nodes, boundary_nodes, elements, boundary_indices):
    """
    Order boundary nodes to form a closed polygon.
    
    Args:
        nodes: Full array of node coordinates
        boundary_nodes: Array of boundary node coordinates
        elements: Connectivity information
        boundary_indices: Indices of boundary nodes in the full node list
        
    Returns:
        ordered_nodes: Ordered array of boundary node coordinates
    """
    # Create a mapping from node index to its position in boundary_indices
    index_to_pos = {idx: i for i, idx in enumerate(boundary_indices)}
    
    # Create a graph of connections between boundary nodes
    connections = defaultdict(set)
    
    # For TRI_3 elements (triangles), each edge connects two nodes
    for elem in elements['TRI_3']:
        for i in range(3):
            n1, n2 = elem[i], elem[(i+1)%3]
            # If both nodes are on the boundary, add their connection
            if n1 in index_to_pos and n2 in index_to_pos:
                connections[n1].add(n2)
                connections[n2].add(n1)
    
    # Start with any boundary node
    start_node = boundary_indices[0]
    ordered_indices = [start_node]
    
    # Find the next node until we've added all boundary nodes
    while len(ordered_indices) < len(boundary_indices):
        current = ordered_indices[-1]
        # Get the connected nodes that are on the boundary
        next_candidates = connections[current]
        
        # Choose the next node that hasn't been visited yet
        next_node = None
        for node in next_candidates:
            if node not in ordered_indices:
                next_node = node
                break
        
        if next_node is None:
            # If we can't find a next node, the boundary may not be a single closed loop
            print("Warning: Boundary may not be a single closed loop")
            break
        
        ordered_indices.append(next_node)
    
    # Create ordered node coordinates using the correct indices
    ordered_positions = np.array([nodes[idx] for idx in ordered_indices])
    
    return ordered_positions





if __name__ == "__main__":
    
    dataset = Dataset()
    problem = ProblemDefinition()

    problem._load_from_dir_(os.path.join('/scratch/dmsm/gi.catalani/2D_profile','problem_definition'))
    dataset._load_from_dir_(os.path.join('/scratch/dmsm/gi.catalani/2D_profile','dataset'), verbose = True)

    print("problem =", problem)
    print("dataset =", dataset)
    
    import pyoche as pch
    # Process training samples
    ids_train = problem.get_split('train')
    for idx in ids_train:
        sample = dataset[idx]
        
        # Get coordinates and compute SDF
        coordinates = sample.get_nodes(base_name="Base_2_2")
        print('Computing SDF for sample Train', idx)
        sdf_values = compute_airfoil_sdf(sample)
        
        # Create sample dictionary with coordinates
        s = pch.Sample.from_dict({
            'point_cloud_field/coordinates': coordinates.T,
        })
        
        # Add SDF to point_cloud_field
        s['point_cloud_field/sdf'] = sdf_values[np.newaxis]
        
        # Add output fields as requested
        for fn in ['Mach', 'Pressure', 'Velocity-x', 'Velocity-y']:
            try:
                field_data = sample.get_field(fn, base_name="Base_2_2")
                s[f'point_cloud_field/{fn}'] = field_data[np.newaxis]
            except (TypeError, KeyError) as e:
                print(f"Could not get field {fn}: {e}")
        
        # Add scalar values
        for sn in sample.get_scalar_names():
            try:
                s[f'scalars/{sn}'] = sample.get_scalar(sn)[np.newaxis]
            except TypeError:
                print(f'Skip scalar {sn}')
        
        # Save the sample
        output_dir = '/scratch/dmsm/gi.catalani/2D_profile/pyoche_data/2D_profile_train.pch'
        os.makedirs(output_dir, exist_ok=True)
        s.save_h5file(f'{output_dir}/_sample_{idx:03d}.h5')

    # Process test samples
    ids_test = problem.get_split('test')
    for idx in ids_test:
        sample = dataset[idx]
        
        # Get coordinates and compute SDF
        coordinates = sample.get_nodes(base_name="Base_2_2")
        print('Computing SDF for sample Train', idx)
        sdf_values = compute_airfoil_sdf(sample)
        
        # Create sample dictionary with coordinates
        s = pch.Sample.from_dict({
            'point_cloud_field/coordinates': coordinates.T,
        })
        
        # Add SDF to point_cloud_field
        s['point_cloud_field/sdf'] = sdf_values[np.newaxis]
        
        # Add output fields as requested
        for fn in ['Mach', 'Pressure', 'Velocity-x', 'Velocity-y']:
            try:
                field_data = sample.get_field(fn, base_name="Base_2_2")
                s[f'point_cloud_field/{fn}'] = field_data[np.newaxis]
            except (TypeError, KeyError) as e:
                print(f"Could not get field {fn}: {e}")
        
        # Add scalar values
        for sn in sample.get_scalar_names():
            try:
                s[f'scalars/{sn}'] = sample.get_scalar(sn)[np.newaxis]
            except TypeError:
                print(f'Skip scalar {sn}')
        
        # Save the sample
        output_dir = '/scratch/dmsm/gi.catalani/2D_profile/pyoche_data/2D_profile_test.pch'
        os.makedirs(output_dir, exist_ok=True)
        s.save_h5file(f'{output_dir}/_sample_{idx:03d}.h5')
