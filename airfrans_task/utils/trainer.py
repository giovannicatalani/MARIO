from functools import partial
from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm 
import torch
from pathlib import Path

from torch_geometric.loader import DataLoader

from utils.device import DEVICE
from utils.callbacks import CallbackList

class Trainer:
    def __init__(self, model, device=DEVICE):
        self.loss = None
        self.val_loss = None

        self.device = device

        self.model = model.to(self.device)

        self.epochs = None
        self.epoch = 0
        self.optimizer = None
        self.lr_scheduler = None
        self.callbacks = []
        self.cfg = None
        self.train_dataloader = None
        self.val_dataloader = None
    

    def train(self, cfg, loss, train_dataset,
              val_dataset=None, train_step_fn=None, callbacks=None):

        self.loss = loss
        
        self.callbacks = (
            CallbackList([]) if callbacks 
            is None else CallbackList(callbacks)
        )

        if not isinstance(cfg, DictConfig):
            cfg = DictConfig(cfg)
        self.cfg = cfg

        if train_step_fn is not None:
            self.step = partial(train_step_fn, self)
        else:
            if hasattr(cfg, 'train_step_fn'):
                self.step = partial(instantiate(cfg.train_step_fn), self)

        self.epochs = cfg.epochs
        self.optimizer = instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.train_dataloader, self.train_dataset = self.get_loader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True, 
            pin_memory=True, num_workers=4, prefetch_factor=4
        )

        if val_dataset is not None:
            self.val_dataloader, self.val_dataset = self.get_loader(
                val_dataset, batch_size=cfg.batch_size, pin_memory=True,
                num_workers=4, prefetch_factor=4
            )

            self.val_loss = self.loss.make_copy_instance()
        
        self.callbacks.on_train_begin(self)

        for epoch in tqdm(range(self.epochs), 
                          total=self.epochs, 
                          desc='Training, epochs'):
            self.epoch = epoch

            self.callbacks.on_epoch_begin(self)

            for batch in self.train_dataloader:
                batch = batch.to(DEVICE)
                self.callbacks.on_batch_begin(self, batch)

                self.optimizer.zero_grad()
                cb_yhat = self.step(batch)

                self.callbacks.before_optimizer_step(self, batch, cb_yhat)
                self.optimizer.step()

                self.callbacks.on_batch_end(self, batch, cb_yhat)
            
            self.callbacks.on_epoch_end(self, batch, cb_yhat)

            if val_dataset is not None:
                with torch.no_grad():
                    self.model.eval()
                    for batch in self.val_dataloader:
                        batch = batch.to(DEVICE)
                        cb_yhat = self.step(batch, is_train=False)

                        self.callbacks.on_val_batch_end(self, batch, cb_yhat)

                    self.callbacks.on_val_epoch_end(self, batch, cb_yhat)
                    del cb_yhat

        self.callbacks.on_train_end(self)
        return self.model
            
    def step(self, batch, is_train=True):
        inputs, outputs = batch 

        y = outputs[0]
        yhat = self.model.forward(*inputs)[0]

        if is_train:
            if self.loss.requires_model_params:
                self.loss(yhat, y, self.model.parameters())
            else:
                self.loss(yhat, y)
            self.loss.main.backward()
        else:
            self.val_loss(yhat, y)
        
        return yhat[:self.cb_batch_size].detach().cpu()

    def load_pretrained(self, model):
        if Path(model).is_file():
            model = torch.load(model, map_location=self.device)

        self.model.load_state_dict(model.state_dict(), strict=False)

    @property
    def cb_batch_size(self):
        if self.cfg.cb_max_batch is None:
            return self.cfg.batch_size
        if self.cfg.cb_max_batch < self.cfg.batch_size:
            return self.cfg.cb_max_batch
        else:
            return self.cfg.batch_size

    def get_loader(self, dataset, **kwargs):
        pytorch_dset = instantiate(self.cfg.dataloader, pyoche_dataset=dataset)
        if hasattr(pytorch_dset, 'collate_fn'):
            kwargs['collate_fn'] = partial(pytorch_dset.collate_fn, pytorch_dset)
        return DataLoader(
            pytorch_dset, 
            **kwargs
        ), pytorch_dset