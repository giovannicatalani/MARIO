from Muscat.Bridges.CGNSBridge import CGNSToMesh, MeshToCGNS, printTree

from plaid.containers.dataset import Dataset
from plaid.problem_definition import ProblemDefinition

from plaid.containers.sample import Sample

from Muscat.Containers import MeshModificationTools as MMT
from Muscat.FE.FETools import PrepareFEComputation
from Muscat.Containers.MeshFieldOperations import GetFieldTransferOp
from Muscat.FE.Fields.FEField import FEField
from Muscat.Containers.Filters import FilterObjects as FO



from Muscat.IO import XdmfWriter as XW

import os
import numpy as np




def project_fields(mesh, output_nodes):
    '''
    return the distance from each output_nodes to the boundary of the mesh
    '''
    MMT.ComputeSkin(mesh, md=None, inPlace=True, skinTagName="Skin")
    ef = FO.ElementFilter(dimensionality=1, nTag = "Holes")



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
















