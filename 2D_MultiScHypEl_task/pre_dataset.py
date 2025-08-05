import numpy as np
import torch, hydra


from hydra.utils import instantiate, get_class

from base_sample_to_geometric import base_sample_to_geometric


from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.utils.split import split_dataset
from plaid.problem_definition import ProblemDefinition

from functools import partial
from omegaconf import DictConfig
from hydra.utils import instantiate


from tqdm import tqdm
import torch
from pathlib import Path

from torch_geometric.loader import DataLoader



import matplotlib.pyplot as plt
import os

from sdf import add_sdf

from datasets import load_dataset

from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.Containers import MeshCreationTools as MCT
from plaid.bridges import huggingface_bridge
from datasets import load_from_disk
from loaderA import elasto_plasto_dynamics_sample_to_geometric
from datasets import load_dataset



def pre_process_dataset():



    hf_dataset = load_dataset("PLAID-datasets/2D_Multiscale_Hyperelasticity")
    dataset_2, problem_2 = huggingface_bridge.huggingface_dataset_to_plaid(hf_dataset)

    Hyper_Elastic_2D = dataset_2




    probleme_def = problem_2
    add_sdf(Hyper_Elastic_2D)




    print("#Splitting Dataset")
    options = {
        'shuffle': False,
        'split_sizes': {
            'test': 376,

        },
    }



    split = split_dataset(Hyper_Elastic_2D, options)

 




    test_data=Hyper_Elastic_2D.get_samples(split['test'])

    train_data=Hyper_Elastic_2D.get_samples(split['other'])


    train_data_geometric={key : base_sample_to_geometric(sample,key,probleme_def) for key, sample in train_data.items()}
    test_data_geometric={key : base_sample_to_geometric(sample,key,probleme_def) for key, sample in test_data.items()}


    for sample in train_data_geometric.values():


        # Input Scalars
        sample['C11'] = sample.input_scalars[0]
        sample['C12'] = sample.input_scalars[1]
        sample['C22'] = sample.input_scalars[2]

        # Output Scalars
        sample['effective_energy'] = sample.output_scalars[0]

        # Output Fields
        sample['u1'] = sample.output_fields[:,0]
        sample['u2'] = sample.output_fields[:,1]
        sample['P11'] = sample.output_fields[:,2]
        sample['P12'] = sample.output_fields[:,3]
        sample['P22'] = sample.output_fields[:,4]
        sample['P21'] = sample.output_fields[:,5]
        sample['psi'] = sample.output_fields[:,6]


    for sample in test_data_geometric.values():


        # Input Scalars
        sample['C11'] = sample.input_scalars[0]
        sample['C12'] = sample.input_scalars[1]
        sample['C22'] = sample.input_scalars[2]



    return(train_data_geometric,test_data_geometric)




