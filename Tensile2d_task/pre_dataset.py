import numpy as np
import torch

from plaid.containers.sample import Sample
from plaid.utils.split import split_dataset
from plaid.problem_definition import ProblemDefinition
from plaid.bridges import huggingface_bridge

from Muscat.Bridges.CGNSBridge import CGNSToMesh
from Muscat.Containers import MeshModificationTools as MMT
from Muscat.FE.FETools import PrepareFEComputation
from Muscat.Containers.MeshFieldOperations import GetFieldTransferOp
from Muscat.FE.Fields.FEField import FEField
from Muscat.Containers.Filters import FilterObjects as FO

from torch_geometric.data import Data
from torch_geometric.utils._coalesce import coalesce as geometric_coalesce

from datasets import load_dataset

def pre_process_dataset():

    hf_dataset = load_dataset("PLAID-datasets/Tensile2d")
    dataset_2, problem_2 = huggingface_bridge.huggingface_dataset_to_plaid(hf_dataset)

    Tensil_2D = dataset_2
    probleme_def = problem_2
    
    add_sdf(Tensil_2D)

    print("#Splitting Dataset")
    ids_train = probleme_def.get_split('train_500')
    ids_test  = probleme_def.get_split('test')

    train_data=Tensil_2D.get_samples(ids_train)
    test_data=Tensil_2D.get_samples(ids_test)

   
    train_data_geometric={key : base_sample_to_geometric(sample,key,probleme_def) for key, sample in train_data.items()}
    test_data_geometric={key : base_sample_to_geometric(sample,key,probleme_def) for key, sample in test_data.items()}

    for sample in train_data_geometric.values():

        # Input Scalars
        sample['p'] = sample.input_scalars[0]
        sample['p1'] = sample.input_scalars[1]
        sample['p2'] = sample.input_scalars[2]
        sample['p3'] = sample.input_scalars[3]
        sample['p4'] = sample.input_scalars[4]
        sample['p5'] = sample.input_scalars[5]

        # Output Scalars
        sample['max_von_mises'] = sample.output_scalars[0]
        sample['max_q'] = sample.output_scalars[1]
        sample['max_U2_top'] = sample.output_scalars[2]
        sample['max_sig22_top'] = sample.output_scalars[3]

        # Output Fields
        sample['U1'] = sample.output_fields[:,0]
        sample['U2'] = sample.output_fields[:,1]
        sample['q'] = sample.output_fields[:,2]
        sample['sig11'] = sample.output_fields[:,3]
        sample['sig22'] = sample.output_fields[:,4]
        sample['sig12'] = sample.output_fields[:,5]

    for sample in test_data_geometric.values():

        # Input Scalars
        sample['p'] = sample.input_scalars[0]
        sample['p1'] = sample.input_scalars[1]
        sample['p2'] = sample.input_scalars[2]
        sample['p3'] = sample.input_scalars[3]
        sample['p4'] = sample.input_scalars[4]
        sample['p5'] = sample.input_scalars[5]


    return(train_data_geometric,test_data_geometric)


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
        input_fields = np.concatenate((vertices, input_fields), axis=1)
        input_fields_names = ["x", "y", *input_fields_names]
    else:
        sdf = sample.get_field("sdf",base_name="Base_2_2")
        sdf = sdf.reshape(-1,1)
        input_fields = result = np.hstack((vertices, sdf))
        input_fields_names = ["x", "y", "sdf"]

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





def faces_to_edges(faces: np.ndarray, num_nodes: int, coalesce: bool=True):
    """Creates a list of edges from a Faces array

    Args:
        faces (np.ndarray): Array of faces shape (n_faces, face_dim)

    Returns:
        np.ndarray: the edge list of shape (n, 2)
    """

    assert len(faces.shape)==2, "Wrong shape for the faces, should be a 2D array"

    # Generate edges (without duplicates in one pass)
    rolled = np.roll(faces, -1, axis=1)
    edges = np.vstack((faces.ravel(), rolled.ravel())).T
    edges = np.concatenate((edges, edges[:, ::-1]), axis=0)

    # Ensure unique edges by sorting each edge and using np.unique
    if coalesce:
        edges = my_coalesce(edges, num_nodes)
    # edges = np.sort(edges, axis=1)
    # edges = np.unique(edges, axis=0)

    return edges


def my_coalesce(edges: torch.Tensor | np.ndarray, num_nodes: int, reduce="add"):
    if isinstance(edges, np.ndarray):
        edges = torch.tensor(edges).T
        return geometric_coalesce(edges, num_nodes=num_nodes, reduce=reduce).T.numpy()
    edges = geometric_coalesce(edges.T, num_nodes=num_nodes, reduce=reduce).T
    return edges


def project_fields(mesh, output_nodes):
    '''
    return the distance from each output_nodes to the boundary of the mesh
    '''
    MMT.ComputeSkin(mesh, md=None, inPlace=True, skinTagName="Skin")
    ef = FO.ElementFilter(dimensionality=1)


    # dim = int(mesh.GetElementsDimensionality())
    space_, Tnumberings, _, _ = PrepareFEComputation(mesh,numberOfComponents=1)
    field_mesh = FEField("", mesh=mesh, space=space_, numbering=Tnumberings[0])

    op, _, _ = GetFieldTransferOp(inputField = field_mesh, targetPoints = output_nodes, method = "Interp/Clamp",
                                    elementFilter= ef, verbose = False)

    pos = op.dot(mesh.nodes)
    sdf = np.sqrt(np.sum((pos - output_nodes)**2, axis=1))
    return sdf


def add_sdf(dataset):
    for sample in dataset:
        tree = sample.get_mesh()
        mesh = CGNSToMesh(tree)
        sdf = project_fields(mesh, mesh.nodes)
        sample.add_field('sdf',sdf,base_name="Base_2_2")