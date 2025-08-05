from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
try:
    from utils.conditioning import shift_modulation
except:
    from conditioning import shift_modulation

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
        modulate_scale=False,
        modulate_shift=True,
        frequency_embedding="nerf",
        include_input=True,
        scale=5,
        max_frequencies=32,
        base_frequency=1.25,
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
        if modulate_scale and modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            self.num_modulations *= 2
        self.latent_to_modulation = LatentToModulation(
            self.latent_dim, self.num_modulations, dim_hidden=128, num_layers=self.depth_hnn
        )
        
        self.conditioning = shift_modulation

    def modulated_forward(self, x, z):
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        #print('x', x.shape, z.shape)
        features = self.latent_to_modulation(z)
        position = self.embedding(x)
        if self.frequency_embedding == "gaussian" and self.include_input:
            position = torch.cat([position, x], axis=-1)
        pre_out = self.conditioning(position, features, self.layers[:-1], torch.relu)
        out = self.layers[-1](pre_out)
        return out.view(*x_shape, out.shape[-1])


    
class MultiScaleModulatedFourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim=5,
        output_dim=1,
        num_frequencies=8,
        latent_dim= 4,
        width=256,
        depth=3,
        depth_hnn = 3,
        include_input=True,
        scales=[1,5],
        conditioning_type='shift_modulation',
        num_heads=4,
        scalar_hidden_dim=128,
        scalar_out_dim=6,
    ):
        
        super().__init__()
       
        self.include_input = include_input
        self.scales = scales
        self.conditioning_type = conditioning_type

        self.embeddings = nn.ModuleList([GaussianEncoding(embedding_size=num_frequencies * 2, scale=scale, dims=input_dim) for scale in scales])
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
        self.num_modulations = self.hidden_dim * (self.depth - 1)
        
        self.cond_to_modulation = LatentToModulation(self.latent_dim, self.num_modulations, dim_hidden=256, num_layers=self.depth_hnn)
        # Choose conditioning type
        if self.conditioning_type == 'shift_modulation':
            self.conditioning = shift_modulation
        else:
            raise ValueError("Invalid conditioning_type. Choose 'shift_modulation' or 'cross_attention'.")
        
        self.scalar_head = nn.Sequential(
            nn.Linear(width * (depth - 1), scalar_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(scalar_hidden_dim, scalar_out_dim),
        )

    def modulated_forward(self, x, z):
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
       
        features = self.cond_to_modulation(z) 
        positions = [embedding(x) for embedding in self.embeddings]

        if self.include_input:
            positions = [torch.cat([pos, x], axis=-1) for pos in positions]
        
        if self.conditioning_type == 'shift_modulation':   
            pre_outs = [self.conditioning(pos, features, self.layers[:-1], torch.relu) for pos in positions]
        outs = [self.layers[-1](pre_out) for pre_out in pre_outs]

        # Concatenate the outputs from each scale
        concatenated_out = torch.cat(outs, axis=-1)

        # A final linear layer to combine multi-scale outputs
        final_out = self.final_linear(concatenated_out)

        return final_out.view(*x_shape, final_out.shape[-1])
    
    def predict_scalars(self, z):
        """
        cond: (batch_size, latent_dim) graph‐level condition vectors
        returns: (batch_size, scalar_out_dim)
        """
        # run hyper‐network on each cond vector
        mod_feats = self.cond_to_modulation(z)      # (batch_size, M)
        return self.scalar_head(mod_feats)

if __name__ == "__main__":
    # Input parameters
    input_dim = 5         # Dimensionality of input coordinates
    latent_dim = 4        # Latent dimension for modulation
    num_frequencies = 8   # Number of Fourier frequencies
    output_dim = 1        # Output dimension
    batch_size = 2        # Number of examples in a batch
    n_samples = 10        # Number of samples per batch
    conditioning_type = 'cross_attention'  # You can switch to 'shift_modulation' to test shift modulation

    # Initialize model
    model = MultiScaleModulatedFourierFeatures(
        input_dim=input_dim,
        output_dim=output_dim,
        num_frequencies=num_frequencies,
        latent_dim=latent_dim,
        conditioning_type=conditioning_type
    )

    # Random input data (positions and latent features)
    positions = torch.randn(batch_size * n_samples, input_dim)  # Random positions
    latent_features = torch.randn(batch_size* n_samples, latent_dim)       # Random latent features

    # Forward pass through the model
    output = model.modulated_forward(positions, latent_features)

    # Print output
    print("Output shape:", output.shape)
    print("Output:", output)