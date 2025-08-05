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
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.Containers import MeshCreationTools as MCT
from plaid.bridges import huggingface_bridge
from datasets import load_dataset




def pre_process_dataset():


    hf_dataset = load_dataset("PLAID-datasets/Tensile2d")
    dataset_2, problem_2 = huggingface_bridge.huggingface_dataset_to_plaid(hf_dataset)




    probleme_def = problem_2



    Tensil_2D = dataset_2


    add_sdf(Tensil_2D)






    print("#Splitting Dataset")
    options = {
        'shuffle': False,
        'split_sizes': {
            'train': 500,    

        },
    }



    ids_train = probleme_def.get_split('train_500')
    ids_test  = probleme_def.get_split('test')




    split = split_dataset(Tensil_2D, options)







    test_data=Tensil_2D.get_samples(ids_test)

    train_data=Tensil_2D.get_samples(ids_train)


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




