import torch
from utils.models import  MultiScaleModulatedFourierFeatures, ModulatedFourierFeatures


def create_inr_instance(cfg, input_dim=1, output_dim=1, device="cuda"):
    device = torch.device(device)

    if cfg.inr.model_type == "ms_fourier_features":
        inr = MultiScaleModulatedFourierFeatures(
            input_dim=input_dim,
            output_dim=output_dim,
            num_frequencies=cfg.inr.num_frequencies,
            latent_dim=cfg.inr.latent_dim,
            width=cfg.inr.hidden_dim,
            depth=cfg.inr.depth,
            depth_hnn= cfg.inr.hypernet_depth,
            include_input=cfg.inr.include_input,
            scales=cfg.inr.scale,
            conditioning_type= cfg.inr.conditionning_type,
            scalar_out_dim= cfg.inr.out_scalar_dim
        ).to(device)

    elif cfg.inr.model_type == "fourier_features":
        inr = ModulatedFourierFeatures(
            input_dim=input_dim,
            output_dim=output_dim,
            num_frequencies=cfg.inr.num_frequencies,
            latent_dim=cfg.inr.latent_dim,
            width=cfg.inr.hidden_dim,
            depth=cfg.inr.depth,
            depth_hnn= 1,
            modulate_scale=cfg.inr.modulate_scale,
            modulate_shift=cfg.inr.modulate_shift,
            frequency_embedding=cfg.inr.frequency_embedding,
            include_input=cfg.inr.include_input,
            scale=cfg.inr.scale,
            max_frequencies=cfg.inr.max_frequencies,
            base_frequency=cfg.inr.base_frequency,
        ).to(device)

    return inr

# Function to load a model
def load_inr(model_weights, cfg,input_dim, output_dim, device="cuda"):
    # ... load the INR model from model_path using the configuration cfg
    # load inr weights

    inr_in = create_inr_instance(cfg, input_dim=input_dim, output_dim=output_dim, device=device)
    inr_in.load_state_dict(model_weights)
    inr_in.eval()

    return inr_in


