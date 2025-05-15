import numpy as np
from tqdm import tqdm
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

import io
from PIL import Image
from pathlib import Path

from utils.device import DEVICE

from utils.funcs import assign_normals, plot_tricontourf, make_bl_layer, compute_airfoil_metrics

import csv
import os

def log_to_csv(filename, epoch, metrics, log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_path = log_dir / filename
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["epoch"] + list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        row = {"epoch": epoch}
        row.update(metrics)
        writer.writerow(row)

def to_PIL(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

class CallbackList(list):
    def __getattr__(self, name):
        def handler(*args, **kwargs):
            for cb in self:
                method = getattr(cb, name, None)
                if callable(method):
                    method(*args, **kwargs)
        return handler

class Callback:

    def on_train_begin(self, trainer, /, *args, **kwargs):
        pass
    def on_train_end(self, trainer, /, *args, **kwargs):
        pass

    def on_epoch_begin(self, trainer, /, *args, **kwargs):
        pass
    def on_epoch_end(self, trainer, /, batch, yhat, *args, **kwargs):
        pass

    def on_batch_begin(self, trainer, /, batch, *args, **kwargs):
        pass
    def on_batch_end(self, trainer, /, batch, yhat, *args, **kwargs):
        pass

    def on_val_epoch_end(self, trainer, /, batch, yhat, *args, **kwargs):
        pass
    def on_val_batch_end(self, trainer, /, batch, yhat, *args, **kwargs):
        pass

    def before_optimizer_step(self, trainer, /, batch, yhat, *args, **kwargs):
        pass

class DataCallback:

    def after_dataset_load(self, *args, **kwargs):
        pass
    def after_train_dataset_load(self, *args, **kwargs):
        pass
    def after_val_dataset_load(self, *args, **kwargs):
        pass
    def after_train_dataset_normalize(self, *args, **kwargs):
        pass
    def after_val_dataset_normalize(self, *args, **kwargs):
        pass

    def before_predict(self, *args, **kwargs):
        pass

#######################################################
# Data Callbacks

class FilterSample(DataCallback):
    """
    We won't ask question, remove the outliers if the validation loss
    goes down its a win.
    """
    def __init__(self, sample_name_list):
        if isinstance(sample_name_list, str):
            raise TypeError('Should be a list')
        self.sample_names = sample_name_list
    
    def after_train_dataset_load(self, dataset, *args, **kwargs):
        dataset.update(
            list(filter(
                lambda x: Path(x.name).stem not in self.sample_names, 
                dataset.samples
            ))
        )

    def after_val_dataset_load(self, dataset, *args, **kwargs):
        dataset.update(
            list(filter(
                lambda x: Path(x.name).stem not in self.sample_names, 
                dataset.samples
            ))
        )
        
class ThickCambCondition(DataCallback):
    def __init__(self):
        pass
    
    def _assign_airfoil_metrics(self, dataset):
        sample_list = []
        for i, sample in tqdm(enumerate(dataset), total=len(dataset), 
                              desc='Assigning Geom Conditions'):

            sample['scalars/airfoil_metrics'] = compute_airfoil_metrics(
                sample
            )
            sample_list.append(sample)
        dataset.update(sample_list)

    def after_train_dataset_load(self, dataset, *args, **kwargs):
        self._assign_airfoil_metrics(dataset)

    def after_val_dataset_load(self, dataset, *args, **kwargs):
        self._assign_airfoil_metrics(dataset)

class ProcessNormals(DataCallback):
    def __init__(self, fade = None, flow_relative = False):
        self.fade = fade
        self.flow_relative = flow_relative
    
    def _assign_normals(self, dataset):
        for i, sample in tqdm(enumerate(dataset), total=len(dataset), 
                              desc='Assigning normals'):
            if self.fade is not None:
                fade_value = self.fade
                fade_bool = True
            else: 
                fade_value = np.inf
                fade_bool = False

            c1 = ((
                sample['point_cloud_field/distance_function'] < fade_value
            )&~sample['point_cloud_field/is_surf'].astype(bool))[0]

            sample = assign_normals(sample, c1, fade=fade_bool, flow_relative=False)
            dataset.samples[i] = sample

    def after_train_dataset_load(self, dataset, *args, **kwargs):
        self._assign_normals(dataset)

    def after_val_dataset_load(self, dataset, *args, **kwargs):
        self._assign_normals(dataset)
    
    def before_predict(self, dataset, *args, **kwargs):
        self._assign_normals(dataset)

class BoundaryLayerInput(DataCallback):
    def __init__(self, thickness, decay='squared'):
        self.thickness = thickness
        self.decay = decay
    
    def _assign_bl_layer(self, dataset):
        sample_list = []
        for sample in tqdm(
            dataset, total=len(dataset), desc='Assigning BL layer'
        ):
            sample['point_cloud_field/bl_layer'] = make_bl_layer(
                sample, self.thickness, decay=self.decay
            )
            sample_list.append(sample)
        dataset.update(sample_list)
        
    def after_train_dataset_load(self, dataset, *args, **kwargs):
        self._assign_bl_layer(dataset)

    def after_val_dataset_load(self, dataset, *args, **kwargs):
        self._assign_bl_layer(dataset)
    
    def before_predict(self, dataset, *args, **kwargs):
        self._assign_bl_layer(dataset)

class PrintFieldStats(DataCallback):
    def __init__(self, features, sample_number=None, after_load=True, after_norm=False):
        self.sample_number = sample_number
        self.features = features
        self.after_load = after_load
        self.after_norm = after_norm

    def after_train_dataset_load(self, dataset, *args, **kwargs):
        if not self.after_load:
            return
        self._print_stats(dataset)
    
    def after_train_dataset_normalize(self, dataset, *args, **kwargs):
        if not self.after_norm:
            return
        self._print_stats(dataset)

    def _print_stats(self, dataset):
        if self.sample_number is None:
            data = dataset
        else:
            data = dataset[self.sample_number]
        for feature in self.features:
            print(f'{feature} with shape {data[feature].shape}')
            print(
                f'\tmin: {data[feature].min()}\n'
                f'\tmax: {data[feature].max()}\n'
                f'\tmean: {data[feature].mean()}'
            )

#######################################################
# Training Callbacks

class SaveModel(Callback):
    def __init__(self, save_path, every_n_epoch=1):
        self.save_path = save_path
        self.n_epoch = every_n_epoch
    
    def on_train_begin(self, trainer, /, *args, **kwargs):
        if 'multirun' in (output_dir:=HydraConfig.get().runtime.output_dir):
            self.save_path = f'{output_dir}/model.pt'
        else:
            self.save_path = self.save_path
        
        with open(self.save_path, 'wb'):
            pass # erroring now in case file can't be written/path doesn't exist
    
    def on_epoch_end(self, trainer, /, *args, **kwargs):
        if not trainer.epoch%self.n_epoch == 0:
           return  
        torch.save(trainer.model, self.save_path)

class LRScheduler(Callback):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def on_train_begin(self, trainer, /, *args, **kwargs):
        trainer._cb_lr_scheduler = instantiate(
            self.kwargs['scheduler'], optimizer = trainer.optimizer
        )

    def on_epoch_end(self, trainer, /, *args, **kwargs):
        trainer._cb_lr_scheduler.step(trainer.loss.get('total_loss'))

class ClipGradients(Callback):
    def __init__(self, max_norm, 
                 norm_type=2, error_if_nonfinite=False,
                 foreach=None
                 ):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.error_if_nonfinite = error_if_nonfinite
        self.foreach = foreach
    
    def before_optimizer_step(self, trainer, /, *args, **kwargs):
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), self.max_norm, self.norm_type,
            self.error_if_nonfinite, self.foreach
        )

class LogFieldWiseMSE(Callback):
    """
    This is hardcoded for the airfrans case. The point is 
    to log each the mse of each field independently. 
    """
    def __init__(self, field_names, log_dir):
        self.log_dir = log_dir
        self.n_samples = 0
        self.val_n_samples = 0

        for i, name in enumerate(field_names):
            if name.endswith('velocity'):
                field_names[i] = 'point_cloud_field/velocity_x'
                field_names.insert(i+1, 'point_cloud_field/velocity_y')
                break

        self.field_names = field_names

        self.epoch_vals = torch.zeros(len(field_names))
        self.val_epoch_vals = torch.zeros(len(field_names))


    def on_batch_end(self, trainer, /, batch, yhat, *args, **kwargs):
        y = batch.y.cpu()
        self.epoch_vals += torch.sum((y-yhat)**2, dim=0)
        self.n_samples += yhat.shape[0]

    def on_val_batch_end(self, trainer, /, batch, yhat, *args, **kwargs):
        y = batch.y.cpu()
        self.val_epoch_vals += torch.sum((y-yhat)**2, dim=0)
        self.val_n_samples += yhat.shape[0]

    def on_epoch_end(self, trainer, /, *args, **kwargs):
        values = self.epoch_vals/self.n_samples
        metrics = {f'train/{key}': value.item() for key, value in zip(self.field_names, values)}
        log_to_csv("train_metrics.csv", trainer.epoch, metrics, self.log_dir)
        self.epoch_vals = torch.zeros(self.epoch_vals.shape)
        self.n_samples = 0

    def on_val_epoch_end(self, trainer, /, *args, **kwargs):
        values = self.val_epoch_vals/self.val_n_samples
        metrics = {f'val/{key}': value.item() for key, value in zip(self.field_names, values)}
        log_to_csv("val_metrics.csv", trainer.epoch, metrics, self.log_dir)
        self.val_epoch_vals = torch.zeros(self.val_epoch_vals.shape)
        self.val_n_samples = 0

class SurfMSE(Callback):
    def __init__(self, every_n_epoch, field_names, log_dir):
        self.log_dir = log_dir
        self.n_epoch = every_n_epoch

        for i, name in enumerate(field_names):
            if name.endswith('velocity'):
                field_names[i] = 'point_cloud_field/velocity_x'
                field_names.insert(i+1, 'point_cloud_field/velocity_y')
                break

        self.field_names = field_names

    def on_val_epoch_end(self, trainer, /, *args, **kwargs):
        if trainer.epoch % self.n_epoch != 0:
            return 

        sample_mses = torch.zeros(
            len(trainer.val_dataset), len(self.field_names)
        )

        for i in range(len(trainer.val_dataset)):
            batch = trainer.val_dataset.get_full_item(i)

            x = batch.x[batch.surf].to(DEVICE)
            z = batch.z[batch.surf].to(DEVICE)
            y = batch.y[batch.surf].to(DEVICE)

            yhat = trainer.model.forward(x, z)[0]
            sample_mses[i] = torch.mean((y-yhat)**2, dim=0)

        mean_mse = torch.mean(sample_mses, dim=0)

        metrics = {f'surf_full_val/{key}': value.item() for key, value in zip(self.field_names, mean_mse)}
        log_to_csv("surf_val_metrics.csv", trainer.epoch, metrics, self.log_dir)

class LogFullFieldWiseMSE(Callback):
    """
    This is hardcoded for the airfrans case. The point is 
    to log each the mse of each field independently. 
    """
    def __init__(self, every_n_epoch, field_names, log_dir):
        self.log_dir = log_dir
        self.n_epoch = every_n_epoch

        for i, name in enumerate(field_names):
            if name.endswith('velocity'):
                field_names[i] = 'point_cloud_field/velocity_x'
                field_names.insert(i+1, 'point_cloud_field/velocity_y')
                break

        self.field_names = field_names

    def on_val_epoch_end(self, trainer, /, *args, **kwargs):
        if trainer.epoch % self.n_epoch != 0:
            return 

        sample_mses = torch.zeros(
            len(trainer.val_dataset), len(self.field_names)
        )

        for i in range(len(trainer.val_dataset)):
            batch = trainer.val_dataset.get_full_item(i)

            x = batch.x.to(DEVICE)
            z = batch.z.to(DEVICE)
            y = batch.y.to(DEVICE)

            yhat = trainer.model.forward(x, z)[0]
            sample_mses[i] = torch.mean((y-yhat)**2, dim=0)

        mean_mse = torch.mean(sample_mses, dim=0)

        metrics = {f'full_val/{key}': value.item() for key, value in zip(self.field_names, mean_mse)}
        log_to_csv("full_val_metrics.csv", trainer.epoch, metrics, self.log_dir)

class LogLoss(Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def on_epoch_end(self, trainer, /, *args, **kwargs):
        train_loss = trainer.loss.flush_epoch()  # expecting a dict of losses
        log_to_csv("loss_metrics.csv", trainer.epoch, train_loss, self.log_dir)
    
    def on_val_epoch_end(self, trainer, /, *args, **kwargs):
        val_loss = trainer.val_loss.flush_epoch()  # expecting a dict of losses
        log_to_csv("val_loss_metrics.csv", trainer.epoch, val_loss, self.log_dir)
