import torch
import numpy as np
import torch
import torch.nn as nn
from utils.device import DEVICE


def shift_modulation(position, features, layers, activation, with_batch=True):
    """Applies film conditioning (add only) on the network.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """
    feature_shape = features.shape[0]  # features.shape[:-1]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)
    # Maybe add assertion here... but if it errors, your feature_dim size is wrong

    if with_batch:
        features = features.reshape(feature_shape, 1, num_hidden, feature_dim // num_hidden)
    else:
        features = features.reshape(feature_shape, num_hidden, feature_dim // num_hidden)

    h = position

    for i, l in enumerate(layers):
        # Maybe also add another assertion here
        h = activation(l(h) + features[..., i, :])
    return h

class LatentToModulation(nn.Module):
    """Maps a latent vector to a set of modulations.
    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(
        self, latent_dim, num_modulations, dim_hidden, num_layers, activation=nn.SiLU
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.activation = activation

        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)
        else:
            layers = [nn.Linear(latent_dim, dim_hidden), self.activation()]
            if num_layers > 2:
                for i in range(num_layers - 2):
                    layers += [nn.Linear(dim_hidden, dim_hidden),
                               self.activation()]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)

class GaussianEncoding(nn.Module):
    def __init__(self, embedding_size, scale, dims=2, gaussian=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale

        if gaussian:
            bvals = torch.randn(embedding_size // 2, dims) * scale
        else:
            bvals = 2.0 ** torch.linspace(0, scale, embedding_size // 2) - 1

            if dims == 1:
                bvals = bvals[:, None]

            elif dims == 2:
                bvals = torch.stack([bvals, torch.zeros_like(bvals)], dim=-1)
                bvals = torch.cat([bvals, torch.roll(bvals, 1, -1)], dim=0)

            else:
                tmp = (dims - 1) * (torch.zeros_like(bvals),)
                bvals = torch.stack([bvals, *tmp], dim=-1)

                tmp = [torch.roll(bvals, i, -1) for i in range(1, dims)]
                bvals = torch.cat([bvals, *tmp], dim=0)

        avals = torch.ones((bvals.shape[0]))
        self.avals = nn.Parameter(avals, requires_grad=False)
        self.bvals = nn.Parameter(bvals, requires_grad=False)

    def forward(self, tensor) -> torch.Tensor:
        """
        Apply positional encoding to the input.
        """

        return torch.cat(
            [
                self.avals * torch.sin((2.0 * np.pi * tensor) @ self.bvals.T),
                self.avals * torch.cos((2.0 * np.pi * tensor) @ self.bvals.T),
            ],
            dim=-1,
        )

class ModulatedFourierFeatures(nn.Module):
    """WARNING: the code does not support non-graph inputs.
        It needs to be adapted for (batch, num_points, coordinates) format
        The FiLM Modulated Network with Fourier Embedding used for the experiments on Airfrans.
        The code relies on conditoning functions: film, film_linear and film_translate.
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_dim=2,
        output_dim=1,
        num_frequencies=8,
        latent_dim=128,
        width=256,
        depth=3,
        depth_hnn = 1,
        frequency_embedding="nerf",
        include_input=True,
        scale=5,
    ):
        super().__init__()
        self.frequency_embedding = frequency_embedding
        self.include_input = include_input
        
        self.scale = scale
        self.embedding = GaussianEncoding(
            embedding_size=num_frequencies * 2, scale=scale, dims=input_dim
        )
        embed_dim = (
            num_frequencies * 2 + input_dim
            if include_input
            else num_frequencies * 2
        )
        self.in_channels = [embed_dim] + [width] * (depth - 1)

        self.out_channels = [width] * (depth - 1) + [output_dim]
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_channels[k], self.out_channels[k]) for k in range(depth)]
        )
        self.depth = depth
        self.hidden_dim = width
        self.depth_hnn = depth_hnn
        self.num_modulations = self.hidden_dim * (self.depth - 1)

        self.latent_to_modulation = LatentToModulation(
            self.latent_dim, self.num_modulations, dim_hidden=128, num_layers=self.depth_hnn
        )
        
        self.conditioning = shift_modulation

    def modulated_forward(self, x, z):
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])

        features = self.latent_to_modulation(z)
        position = self.embedding(x)
        if self.frequency_embedding == "gaussian" and self.include_input:
            position = torch.cat([position, x], axis=-1)
        pre_out = self.conditioning(position, features, self.layers[:-1], torch.relu)
        out = self.layers[-1](pre_out)
        return out.view(*x_shape, out.shape[-1])

class Model(nn.Module):
    def __init__(self, input_scaler, output_scaler):
        super(Model, self).__init__()
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
    
    def forward(self, X):
        raise NotImplementedError

    def predict(self, X):
        X = self.input_scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float).to(DEVICE)
        yhat = self.forward(X)
        return self.output_scaler.inverse(yhat.detach().cpu().numpy())

class ConditionalModel(Model):
    def __init__(self, input_scaler, output_scaler, cond_scaler, callback_list=None):
        super(ConditionalModel, self).__init__(
            input_scaler, output_scaler
        )
        self.cond_scaler = cond_scaler

        self.cb_list = callback_list
    
    def predict(self, X, z, to_shape=None):
        X = self.input_scaler.transform(X)
        z = self.cond_scaler.transform(z)

        X = torch.tensor(X, dtype=torch.float).to(DEVICE)
        z = torch.tensor(z, dtype=torch.float).to(DEVICE)

        yhat = self.forward(X, z)[0]

        yhat = self.output_scaler.inverse(yhat.detach().cpu().numpy())

        if to_shape is not None:
            return yhat.reshape(to_shape)
        else:
            return yhat
        
    def apply_datacallbacks(self, dataset):
        """apply the datacallbacks used when training the model."""
        self.cb_list.before_predict(dataset)

class Mario(ConditionalModel):
    def __init__(
        self,
        input_scaler, 
        output_scaler,
        cond_scaler,
        input_dim=5,
        output_dim=1,
        num_frequencies=8,
        latent_dim= 4,
        width=256,
        depth=3,
        width_hnn = 256,
        depth_hnn = 1,
        include_input=True,
        scales=[1, 2, 3],
    ):
        
        super(Mario, self).__init__(
            input_scaler, output_scaler, cond_scaler
        )
       
        self.include_input = include_input
        self.scales = scales

        self.embeddings = nn.ModuleList([
            GaussianEncoding(
                embedding_size=num_frequencies * 2, 
                scale=scale, dims=input_dim
            ) 
            for scale in scales
        ])
        embed_dim = num_frequencies * 2 
        embed_dim += input_dim if include_input else 0
        self.in_channels = [embed_dim] + [width] * (depth - 1)

        self.out_channels = [width] * (depth - 1) + [width]
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_channels[k], self.out_channels[k]) for k in range(depth)]
        )
        self.final_linear = nn.Linear(len(self.scales) * width, output_dim)
        self.depth = depth
        self.hidden_dim = width
        self.depth_hnn = depth_hnn
        self.hidden_dim_hnn = width_hnn
        self.num_modulations = self.hidden_dim * (self.depth - 1)
        
        self.cond_to_modulation = LatentToModulation(
            self.latent_dim, self.num_modulations, 
            dim_hidden=self.hidden_dim_hnn, num_layers=self.depth_hnn
        )

        self.conditioning = shift_modulation

    def forward(self, x, z):
        osx = None
        if x.dim() == 3:
            osx = x.shape
            x = torch.flatten(x.permute(0, 2, 1), end_dim=1)
        if z.dim() == 3:
            z = torch.flatten(z.permute(0, 2, 1), end_dim=1)

        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
       
        features = self.cond_to_modulation(z) 
        positions = [embedding(x) for embedding in self.embeddings]

        if self.include_input:
            positions = [torch.cat([pos, x], axis=-1) for pos in positions]
        
        pre_outs = [
            self.conditioning(pos, features, self.layers[:-1], torch.relu) 
            for pos in positions
        ]
        outs = [self.layers[-1](pre_out) for pre_out in pre_outs]

        # Concatenate the outputs from each scale
        concatenated_out = torch.cat(outs, axis=-1)

        # A final linear layer to combine multi-scale outputs
        final_out = self.final_linear(concatenated_out)

        final_out = final_out.view(*x_shape, final_out.shape[-1])

        if osx is not None:
            return (final_out.view(osx[0], final_out.shape[-1], osx[-1]),)
        else: 
            return (final_out,)
        
        
