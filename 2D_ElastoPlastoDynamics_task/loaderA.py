from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from torch_geometric.data import Data
import numpy as np
from faces_to_edge_ import faces_to_edges

import torch

from Muscat.Containers import MeshModificationTools as MMT
from Muscat.Bridges.CGNSBridge import CGNSToMesh
from Muscat.Containers.Filters import FilterObjects as FO
from sklearn.neighbors import KDTree


def elasto_plasto_dynamics_sample_to_geometric(sample: Sample, sample_id: int, problem_definition: ProblemDefinition) -> Data:
    """
    Converts a Plaid sample to PytorchGeometric Data object

    Args:
        sample (plaid.containers.sample.Sample): data sample

    Returns:
        Data: the converted data sample
    """

    vertices = sample.get_vertices()

    edge_index = []
    coalesce = True
    for _, faces in sample.get_elements().items():
        edge_index.append(faces_to_edges(faces, num_nodes=vertices.shape[0], coalesce=coalesce))
    edge_index = np.concatenate(edge_index, axis=0)

    mesh = CGNSToMesh(sample.get_mesh())
    MMT.ComputeSkin(mesh, md = 2, inPlace = True, skinTagName="Skin")
    nfSkin = FO.NodeFilter(eTag = "Skin")
    nodeIndexSkin = nfSkin.GetNodesIndices(mesh)
    mesh.GetNodalTag("Skin").AddToTag(nodeIndexSkin)
    border_ids = mesh.GetNodalTag("Skin").GetIds()

    sdf , _ = get_distance_to_ids(vertices, border_ids)
    sdf = sdf.reshape(-1, 1)
    input_fields = np.concatenate((vertices, sdf), axis=1)
    input_fields_names = ["x", "y", "sdf", "U_x", "U_y"]
    output_fields_names = ["U_x", "U_y"]
    input_scalars_names = ["time"]
    output_scalars_names = []





    timesteps = sample.get_all_mesh_times()
    output_fields_t0 = np.vstack((sample.get_field("U_x", time=timesteps[0]), sample.get_field("U_y", time=timesteps[0]))).T
    output_fields_t2 = np.vstack((sample.get_field("U_x", time=timesteps[0]), sample.get_field("U_y", time=timesteps[0]))).T
    data_list = []





    for t0, t1 in zip(timesteps[:-1], timesteps[1:]):

        output_fields_t1 = output_fields_t2
        output_fields_t2 = np.vstack((sample.get_field("U_x", time=t1), sample.get_field("U_y", time=t1))).T


        input_scalars = np.array([t0])
        if output_fields_t1[0][0] == None :
            input_fields = np.column_stack((vertices, sdf, output_fields_t0))

        else :
            input_fields = np.column_stack((vertices, sdf, output_fields_t1))
        output_fields = output_fields_t2

        # torch tensor conversion
        input_scalars   = torch.tensor(input_scalars, dtype=torch.float32).reshape(1, -1)
        input_fields    = torch.tensor(input_fields, dtype=torch.float32)

        vertices        = torch.tensor(vertices, dtype=torch.float32)
        # edge_weight     = torch.tensor(edge_weight, dtype=torch.float32)
        edge_index      = torch.tensor(edge_index, dtype=torch.long)
        faces           = torch.tensor(faces, dtype=torch.long)

        # Extracting special nodal tags
        nodal_tags = {}
        for k, v in sample.get_nodal_tags().items():
            nodal_tags["border_id"] = torch.tensor(border_ids, dtype=torch.int)

        if None not in output_fields:
            output_fields   = torch.tensor(output_fields, dtype=torch.float32)

            data = Data(
                pos = vertices,
                input_scalars = input_scalars,
                x = input_fields,
                output_fields = output_fields,
                edge_index = edge_index.T,
                # edge_weight = edge_weight,
                faces = faces,
                sample_id = sample_id,
                input_fields_names=input_fields_names,
                output_fields_names=output_fields_names,
                input_scalars_names=input_scalars_names,
                output_scalars_names=output_scalars_names,
                time=t0,
                **nodal_tags
            )
        else:

            data = Data(
                pos = vertices,
                input_scalars = input_scalars,
                x = input_fields,
                edge_index = edge_index.T,
                # edge_weight = edge_weight,
                faces = faces,
                sample_id = sample_id,
                input_fields_names=input_fields_names,
                output_fields_names=output_fields_names,
                input_scalars_names=input_scalars_names,
                output_scalars_names=output_scalars_names,
                time=t0,
                **nodal_tags
            )
        data_list.append(data)

    return data_list




def get_distance_to_ids(vertices, boundary_ids):
        boundary_vertices = vertices[boundary_ids, :]
        search_index = KDTree(boundary_vertices)
        sdf, projection_id = search_index.query(vertices, return_distance=True)

        projection_vertices = boundary_vertices[projection_id.ravel()]
        projection_vectors = (projection_vertices - vertices)
        projection_vectors_norm = np.linalg.norm(projection_vectors, axis=1)
        projection_vectors_norm[projection_vectors_norm==0] = 1
        projection_vectors = projection_vectors / projection_vectors_norm[:, None]

        return sdf, projection_vectors

