from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from torch_geometric.data import Data
import numpy as np
from faces_to_edge_ import faces_to_edges
import torch


def base_sample_to_geometric(sample: Sample, sample_id: int, problem_definition: ProblemDefinition) -> Data:
    """
    Converts a Plaid sample to PytorchGeometric Data object

    Args:
        sample (plaid.containers.sample.Sample): data sample

    Returns:
        Data: the converted data sample
    """

    vertices            = sample.get_vertices(base_name="Base_2_2")
    edge_index = []
    n_elem_types = len(sample.get_elements(base_name="Base_2_2"))
    coalesce = True
    if n_elem_types>1:
        coalesce = False
    assert len(sample.get_elements(base_name="Base_2_2"))==1, "More than one element type"
    for _, faces in sample.get_elements(base_name="Base_2_2").items():
        edge_index.append(faces_to_edges(faces, num_nodes=vertices.shape[0], coalesce=coalesce))
    edge_index = np.concatenate(edge_index, axis=0)
    if not coalesce:
        edge_index = coalesce(edge_index)

    v1 = vertices[edge_index[:, 0]]
    v2 = vertices[edge_index[:, 1]]
    edge_weight = np.linalg.norm(v2 - v1, axis=1)

    # loading scalars
    input_scalars_names     = problem_definition.get_input_scalars_names()
    output_scalars_names    = problem_definition.get_output_scalars_names()

    input_scalars   = []
    output_scalars  = []
    for name in input_scalars_names:
        input_scalars.append(sample.get_scalar(name))
    for name in output_scalars_names:
        output_scalars.append(sample.get_scalar(name))

    # loading fields
    input_fields_names   = problem_definition.get_input_fields_names()
    output_fields_names  = problem_definition.get_output_fields_names()

    if len(input_fields_names) > 0:
        if input_fields_names[0]=="cell_ids":  input_fields_names.pop(0)

    if len(input_fields_names)>=1:
        input_fields    = []
        for field_name in input_fields_names:
            if field_name == 'M_iso':
                pass
                #input_fields.append(sample.get_field(field_name,base_name="Base_1_2"))
            else:
                input_fields.append(sample.get_field(field_name,base_name="Base_2_2"))



        input_fields = np.vstack(input_fields).T
        input_fields = np.concatenate((vertices, input_fields,sample.get_field('sdf',base_name="Base_2_2") ), axis=1)
        input_fields_names = ["x", "y",'sdf', *input_fields_names]
    else:


        sdf = sample.get_field('sdf',base_name="Base_2_2")
        sdf = sdf.reshape(-1,1)


        input_fields = np.hstack((vertices, sdf))

    output_fields   = []
    for field_name in output_fields_names:
        if field_name == 'M_iso':
            pass
            #output_fields.append(sample.get_field(field_name, base_name="Base_1_2"))
        else:
            output_fields.append(sample.get_field(field_name))







    output_fields = np.vstack(output_fields).T

    # torch tensor conversion
    input_scalars   = torch.tensor(input_scalars, dtype=torch.float32)
    input_fields    = torch.tensor(input_fields, dtype=torch.float32)

    vertices        = torch.tensor(vertices, dtype=torch.float32)
    edge_weight     = torch.tensor(edge_weight, dtype=torch.float32)
    edge_index      = torch.tensor(edge_index, dtype=torch.long)
    faces           = torch.tensor(faces, dtype=torch.long)

    # Extracting special nodal tags
    nodal_tags = {}
    for k, v in sample.get_nodal_tags(base_name="Base_2_2").items():
        nodal_tags[k + "_id"] = torch.tensor(v, dtype=torch.long)

    if None not in output_scalars and None not in output_fields:
        output_scalars  = torch.tensor(output_scalars, dtype=torch.float32)
        output_fields   = torch.tensor(output_fields, dtype=torch.float32)

        data = Data(
            pos = vertices,
            input_scalars = input_scalars,
            x = input_fields,
            output_scalars = output_scalars,
            output_fields = output_fields,
            edge_index = edge_index.T,
            edge_weight = edge_weight,
            faces = faces,
            sample_id = sample_id,
            input_fields_names=input_fields_names,
            output_fields_names=output_fields_names,
            input_scalars_names=input_scalars_names,
            output_scalars_names=output_scalars_names,
            **nodal_tags
        )

        return data

    data = Data(
        pos = vertices,
        input_scalars = input_scalars,
        x = input_fields,
        edge_index = edge_index.T,
        edge_weight = edge_weight,
        face_id = faces,
        sample_id = sample_id,
        input_fields_names=input_fields_names,
        output_fields_names=output_fields_names,
        input_scalars_names=input_scalars_names,
        output_scalars_names=output_scalars_names,
        **nodal_tags
    )

    return data
