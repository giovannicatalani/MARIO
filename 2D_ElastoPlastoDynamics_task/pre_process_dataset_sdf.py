import numpy as np
import torch, hydra
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.Containers import MeshCreationTools as MCT
from plaid.bridges import huggingface_bridge
from datasets import load_from_disk
from loaderA import elasto_plasto_dynamics_sample_to_geometric
from datasets import load_dataset


def pre_process_sdf():

    hf_dataset = load_dataset("PLAID-datasets/2D_ElastoPlastoDynamics")
    dataset_2, problem_2 = huggingface_bridge.huggingface_dataset_to_plaid(hf_dataset)

    print("#Splitting Dataset")

    Elasto_PlastoDynamics = dataset_2
    probleme_def = problem_2

    ids_train = probleme_def.get_split('train')
    ids_test  = probleme_def.get_split('test')
    print("SPLIT SPLIT")
    print(ids_train)
    print(ids_test)
    print(len(ids_train))
    print(len(ids_test))



    test_data = Elasto_PlastoDynamics.get_samples(ids_test)
    train_data = Elasto_PlastoDynamics.get_samples(ids_train[:200])


    train_data_geometric = {key : elasto_plasto_dynamics_sample_to_geometric(sample,key,probleme_def) for key, sample in train_data.items()}
    test_data_geometric = {key : elasto_plasto_dynamics_sample_to_geometric(sample,key,probleme_def) for key, sample in test_data.items()}

    return(train_data_geometric, test_data_geometric)