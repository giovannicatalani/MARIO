from abc import ABC
from copy import deepcopy

from torch import nn
from hydra.utils import instantiate

class Component(ABC):
    def __init__(self):
        self.name = None
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class LossComponent(Component):
    def __call__(self, prediction, target):
        raise NotImplementedError

class MSELoss(LossComponent):
    def __init__(self):
        super().__init__()
        self.name = 'mse_loss'
        self.criterion = nn.MSELoss()

    def __call__(self, prediction, target):
        return self.criterion(prediction, target)


class RegularizationComponent(Component):
    def __call__(self, model_params):
        raise NotImplementedError

class L1Regularization(RegularizationComponent):
    def __init__(self):
        super().__init__()
        self.name = 'l1_reg'
    
    def criterion(self, model_params):
        return sum(p.abs().sum() for p in model_params)

    def __call__(self, model):
        return self.criterion(model)

class Loss:
    def __init__(self, *loss_components_with_coefs, prefix='train'):
        self.loss_components = {}
        self.loss_values = {}
        self.prefix = prefix
        self.loss_components_with_coefs = loss_components_with_coefs

        self.n_samples_batch = 0
        self.n_samples_epoch = 0

        for component, _ in loss_components_with_coefs:
            if isinstance(component, RegularizationComponent):
                self.requires_model_params = True
            else:
                self.requires_model_params = False
    
    @classmethod
    def from_hydra(cls, cfg):
        loss_components_with_coefs=[]
        try:
            coefs = cfg.coefficients
        except AttributeError:
            coefs = [1]*len(cfg.components)
        for component, coef in zip(cfg.components, coefs):
            loss_component = instantiate(component)
            loss_components_with_coefs.append((loss_component, coef))

        return cls(*loss_components_with_coefs)

    def __call__(self, prediction, target, model_params=None, n_samples_batch=None):
        """Call for the loss object, to increment all the contained components.
        if loss object contains regularization components, the model_params arg
        should be set.
        if the first axis of the prediction arg doesn't correspond to the number
        of samples, then n_samples_batch should be passed.
        """
        if self.requires_model_params and model_params is None:
            raise ValueError('Model params are required if a reg term is used')
        
        if n_samples_batch is None:
            self.n_samples_batch = prediction.shape[0]
            self.n_samples_epoch += prediction.shape[0]
        else:
            self.n_samples_batch = n_samples_batch
            self.n_samples_epoch += n_samples_batch
        
        total_loss = 0
        for component, coef in self.loss_components_with_coefs:

            if isinstance(component, RegularizationComponent):
                loss_element = component(model_params)
            elif isinstance(component, LossComponent):
                loss_element = component(prediction, target)
            else:
                raise TypeError('Wrong object passed as loss element')

            weighted_loss = coef * loss_element
            total_loss += weighted_loss
            self.increment_component(component.name, loss_element)
        
        self.increment_component('total_loss', total_loss)

    def get(self, key, value=True):
        if value:
            return self.loss_values[f'{self.prefix}/{key}']
        else:
            return self.loss_components[f'{self.prefix}/{key}']

    @property
    def main(self):
        return self.loss_components[f'{self.prefix}/total_loss']

    def flush_epoch(self):
        if self.n_samples_epoch == 0:
            raise ValueError(
                'The loss object must be called at least '
                'once before flushing values')
        logged_losses = {
            key: value / self.n_samples_epoch
            for key, value
            in self.loss_values.items()
        }

        for loss_component, coef in self.loss_components_with_coefs:
            logged_losses[f'{self.prefix}/{loss_component.name}'] *= coef

        self.loss_components = {}
        self.loss_values = {}
        self.n_samples_batch = 0
        self.n_samples_epoch = 0

        return logged_losses
    
    def increment_component(self, name, value):
        self.loss_components[f'{self.prefix}/{name}'] = value / self.n_samples_batch
        try:
            self.loss_values[f'{self.prefix}/{name}'] += value.item()
        except KeyError:
            self.loss_values[f'{self.prefix}/{name}'] = value.item()

    def make_copy_instance(self, prefix='val', remove_reg=True):
        lcwc = list(deepcopy(self.loss_components_with_coefs))

        if remove_reg:
            i = 0
            while i < len(lcwc):
                if isinstance(lcwc[i][0], RegularizationComponent):
                    lcwc.pop(i)
                    continue
                else:
                    i += 1 
                    
        instance = self.__class__(*lcwc, prefix=prefix)
        return instance