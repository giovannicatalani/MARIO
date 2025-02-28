import numpy as np
import torch, hydra
import airfrans as af

from hydra.utils import instantiate, get_class

from pyoche.ml import MlDataset
from pyoche.ml.normalize import scaler_from_keys

from utils.trainer import Trainer
from utils.loss import Loss
from utils.callbacks import CallbackList
from utils.funcs import train_step_function, process_dataset

@hydra.main(version_base="1.3", config_path='.', config_name='config.yaml')
def main(cfg):

    if hasattr(cfg.training, 'seed') and cfg.training.seed is not None:
        torch.manual_seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
##########################################################
# callback init

    datacallbacks = CallbackList(
        [
            instantiate(callback, _recursive_=False) for callback
            in cfg.data.callbacks
        ]
    )

    traincallbacks = [
                instantiate(callback, _recursive_=False) for callback
                in cfg.training.callbacks
            ]
##########################################################
#  Data loading
    #train_data = af.dataset.load(root=cfg.data.folder_path, task='scarce', train=True)
    #val_data = af.dataset.load(root=cfg.data.folder_path, task='scarce', train=False)

    #train_set = MlDataset(process_dataset(train_data))
    #val_set = MlDataset(process_dataset(val_data))

    # if data is saved as pyoche samples (can be done with the save_to_pch.py script)
    train_set = MlDataset.from_folder(cfg.data.folder_path_train)
    val_set = MlDataset.from_folder(cfg.data.folder_path_val)

    datacallbacks.after_train_dataset_load(train_set)
    datacallbacks.after_val_dataset_load(val_set)
##########################################################
# Data normalization

    scaler_dict = {}
    if hasattr(cfg.normalization, 'default'):
        for group, scaler_type in cfg.normalization.default.items():
            for feature in getattr(cfg.data, group):
                scaler_dict[feature] = get_class(scaler_type)()

    if hasattr(cfg.normalization, 'scalers'):
        for feature, scaler_type in cfg.normalization.scalers.items():
            scaler_dict[feature] = get_class(scaler_type)()

    train_set = train_set.fit_normalize(scaler_dict)
    datacallbacks.after_train_dataset_normalize(train_set)

    val_set = val_set.normalize(scaler_dict)
    datacallbacks.after_val_dataset_normalize(val_set)
##########################################################
# Model instantiation
    
    in_scaler = scaler_from_keys(scaler_dict, *cfg.data.input_features)
    out_scaler = scaler_from_keys(scaler_dict, *cfg.data.output_features)
    cond_scaler = scaler_from_keys(
        scaler_dict, *cfg.data.conditioning_features)

    model = instantiate(
        cfg.model, 
        input_dim = train_set.get_shape(*cfg.data.input_features)[-1],
        output_dim = train_set.get_shape(*cfg.data.output_features)[-1],
        latent_dim = train_set.get_shape(*cfg.data.conditioning_features)[-1],
        input_scaler = in_scaler,
        output_scaler = out_scaler,
        cond_scaler = cond_scaler
    )
##########################################################
# Loss and trainer definition, training launch

    loss = Loss.from_hydra(cfg.training.loss)

    trainer = Trainer(model)

    trainer.train(
        cfg.training, 
        loss=loss,
        train_dataset=train_set, 
        val_dataset=val_set,
        train_step_fn=train_step_function,
        callbacks=traincallbacks)
##########################################################

if __name__ == "__main__":
    main()