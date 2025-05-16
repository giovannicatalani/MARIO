import os
import matplotlib.pyplot as plt
import sys
import yaml
from pathlib import Path
from pickletools import OpcodeInfo
import time
sys.path.append(str(Path(__file__).parents[1]))
from tqdm import tqdm
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from utils.load_models import create_inr_instance,load_inr
from torch_geometric.loader import DataLoader
import wandb
from plaid.containers.dataset import Dataset
from plaid.problem_definition import ProblemDefinition
from utils.dataset import VKIFlowDataset, subsample_dataset
from utils.utils_training import training_step
from torch.utils.data import Subset


@hydra.main(config_path="", config_name="config_out.yaml")
def main(cfg: DictConfig) -> None:
    # Initialize wandb
    wandb.init(project='vki_blade', config=OmegaConf.to_container(cfg, resolve=True))
    wandb.config.update({
        "batch_size": cfg.optim.batch_size,
        "lr_inr": cfg.optim.lr_inr,
        "epochs": cfg.optim.epochs,
        "num_points": cfg.optim.num_points,
        "latent_dim": cfg.inr.latent_dim,
        "model_type": cfg.inr.model_type
    })

    torch.set_default_dtype(torch.float32)

    # optim
    batch_size = cfg.optim.batch_size
    lr_inr = cfg.optim.lr_inr 
    epochs = cfg.optim.epochs
    num_points = cfg.optim.num_points
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # inr
    latent_dim = cfg.inr.latent_dim
    in_dim = cfg.inr.in_dim
    out_dim = cfg.inr.out_dim
    out_scalar_dim = cfg.inr.out_scalar_dim
    latent_cond = True
    
    
    # —── Data loading
    dataset = Dataset()
    problem = ProblemDefinition()

    problem._load_from_dir_(os.path.join(cfg.dataset.path,'problem_definition'))
    dataset._load_from_dir_(os.path.join(cfg.dataset.path,'dataset'), verbose = True)
    ids_train = problem.get_split('train')
    ids_test  = problem.get_split('test')
       
    train_latents = np.load(cfg.dataset.train_latents_path)
    test_latents = np.load(cfg.dataset.test_latents_path)
    # Create dataset instance
    # 3) Build the TRAIN-mode dataset (this computes all normalization stats)
    train_ds = VKIFlowDataset(
    dataset, ids_train,
    geom_latents = train_latents['train_modulations'],
    mode         = 'train')
    
    # 4) Randomly split off 50 samples for validation
    num_total = len(train_ds)
    val_size  = 50
    rng = np.random.RandomState(42)
    all_idx = np.arange(num_total)
    rng.shuffle(all_idx)

    val_idx   = all_idx[:val_size].tolist()
    train_idx = all_idx[val_size:].tolist()

    train_dataset = Subset(train_ds, train_idx)
    val_dataset   = Subset(train_ds, val_idx)

    # 4) Split off 50 for validation
    coef_norm = train_ds.coef_norm

    # 5) Build the TEST-mode dataset (reusing the same stats from full_train_ds)
    test_dataset = VKIFlowDataset(
        dataset, ids_test,
        geom_latents = test_latents['val_modulations'],
        mode         = 'test',
        coef_norm    = coef_norm,
    )
    

    ntrain = len(train_dataset)
    nval = len(val_dataset)
    ntest = len(test_dataset)

    inr_in = create_inr_instance(cfg, input_dim=in_dim, output_dim=out_dim, device=device)
    
    optimizer_in = torch.optim.AdamW(
        [
            {"params": inr_in.parameters()},
        ],
        lr=lr_inr,
        weight_decay=0,
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_in,
        mode='min',
        factor=0.99,      # Reduce LR when plateauing
        patience=200,     # Number of epochs to wait before reducing LR
        verbose=True,    # Print message when LR is reduced
        min_lr=1e-5    # Minimum LR threshold
    )
    
    
    # Improved folder naming with timestamp and all output target fields
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    folder_name = f"training_result_{timestamp}" 
    dir = os.path.dirname(__file__)
    results_directory = os.path.join(dir + '/trainings', folder_name)
    os.makedirs(results_directory, exist_ok=True)
    cfg_save_path = os.path.join(results_directory, 'config.yaml')
    # Convert OmegaConf DictConfig to a native Python dictionary
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # Set `resolve=True` to resolve all interpolations
    with open(cfg_save_path, 'w') as file:
        yaml.dump(cfg_dict, file, default_flow_style=False)

    # Generate run name
    run_name = f"{cfg.inr.model_type}_nl{cfg.inr.depth}_np{num_points}_epochs{epochs}_lr{lr_inr}_ldim{latent_dim}"
   
    best_loss = np.inf
    train_loss_history = []
    test_loss_history = []

    # Check if restart training
    if cfg.restart_training:
        if os.path.exists(cfg.saved_model_path):
            checkpoint = torch.load(cfg.saved_model_path)
            inr_in.load_state_dict(checkpoint["inr_in"])
            optimizer_in.load_state_dict(checkpoint["optimizer_inr_in"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Restarting training from epoch {start_epoch}")
        else:
            print("Saved model not found. Starting training from scratch.")
            start_epoch = 0
    else:
        start_epoch = 0
    
    start_time = time.time()
    for step in tqdm(range(start_epoch, epochs)):
        fit_train_mse_in = 0
        fit_test_mse_in = 0
        fit_field  = 0.0
        fit_scalar = 0.0
        fit_test_field = 0.0
        fit_test_scalar = 0.0
        test_loss_in  = 1
        # Initialize lists for loss history
        
        use_pred_loss = step % 5 == 0
        
        # Prepare the training dataset for the new epoch
        # Subsample the training dataset
        train_subsample_dataset = subsample_dataset(train_dataset, num_points)
        train_loader = DataLoader(train_subsample_dataset, batch_size=batch_size, shuffle=True)
        
        print('Num points for training', len(train_subsample_dataset[0].input))
        for substep,graph in enumerate(train_loader): 
             
            n_samples = len(graph)
            inr_in.train()
            graph = graph.cuda() if torch.cuda.is_available() else graph
            
            outputs = training_step(
                inr_in,
                graph,
                latent_cond=latent_cond,
                return_reconstructions=False,
            )

            optimizer_in.zero_grad()
            outputs["loss"].backward(create_graph=False)
            nn.utils.clip_grad_value_(inr_in.parameters(), clip_value=1.0)
            optimizer_in.step()
            loss = outputs["loss"].cpu().detach()
            field_loss = outputs["field_loss"].cpu().detach()
            scalar_loss = outputs["scalar_loss"].cpu().detach()
            fit_train_mse_in += loss.item() * n_samples
            fit_field += field_loss.item() * n_samples
            fit_scalar += scalar_loss.item() * n_samples
            
            
            
        train_loss_in = fit_train_mse_in / (ntrain)
        train_loss_field = fit_field / (ntrain)
        train_loss_scalar = fit_scalar / (ntrain)
        train_loss_history.append({'epoch': step, 'loss': train_loss_in, 'field_loss': train_loss_field, 'scalar_loss': train_loss_scalar})
        # Inside training loop
        wandb.log({"train_loss": train_loss_in, "epoch": step, "train_field_loss": train_loss_field, "train_scalar_loss": train_loss_scalar})
        print('Train Loss', train_loss_in, 'Field Loss', train_loss_field, 'Scalar Loss', train_loss_scalar)
       

        if use_pred_loss:
            # Prepare the training dataset for the new epoch
            val_subsample_dataset = subsample_dataset(val_dataset, num_points)
            val_loader = DataLoader(val_subsample_dataset, batch_size=batch_size, shuffle=True)
            for substep,graph in enumerate(val_loader):  
                n_samples = len(graph)
                inr_in.train()
                graph = graph.cuda() if torch.cuda.is_available() else graph
                outputs = training_step(
                    inr_in,
                    graph,
                    latent_cond=latent_cond,
                    return_reconstructions=False,
                )
                loss = outputs["loss"].cpu().detach()
                fit_test_mse_in += loss.item() * n_samples
                field_test_loss = outputs["field_loss"].cpu().detach()
                scalar_test_loss = outputs["scalar_loss"].cpu().detach()
                fit_train_mse_in += loss.item() * n_samples
                fit_test_field += field_test_loss.item() * n_samples
                fit_test_scalar += scalar_test_loss.item() * n_samples
              
            test_loss_in = fit_test_mse_in / (ntest)
            test_loss_field = fit_test_field / (ntest)
            test_loss_scalar = fit_test_scalar / (ntest)
            
            scheduler.step(test_loss_in)
            
            test_loss_history.append({'epoch': step, 'loss': test_loss_in, 'field_loss': test_loss_field, 'scalar_loss': test_loss_scalar})
            wandb.log({"val_loss": test_loss_in, "epoch": step, "val_field_loss": test_loss_field, "val_scalar_loss": test_loss_scalar})
            print('Test Loss', test_loss_in, 'Field Loss', test_loss_field, 'Scalar Loss', test_loss_scalar)

            plt.figure()
            plt.plot([p['epoch'] for p in train_loss_history], [p['loss'] for p in train_loss_history], label='Train Loss')
            plt.plot([p['epoch'] for p in test_loss_history], [p['loss'] for p in test_loss_history], label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.title('Training and Test Loss')
            plt.legend()
            plt.savefig(os.path.join(results_directory, f"{run_name}_loss_plot.png"))
            plt.close()
        
        if test_loss_in < best_loss:
            best_loss = test_loss_in
            
            torch.save(
                {
                    "cfg": cfg,
                    "epoch": step,
                    "inr_in": inr_in.state_dict(),
                    "optimizer_inr_in": optimizer_in.state_dict(),
                },
                os.path.join(results_directory, f"{run_name}.pt"),
            )
             
    inr_model = load_inr(torch.load(os.path.join(results_directory, f"{run_name}.pt"))["inr_in"], cfg, in_dim, out_dim, device=device)
    inr_model.eval()
        # —————————— TEST‐TIME INFERENCE & COLLECTION ——————————
    # Prepare containers
    all_field_preds   = []
    all_scalar_preds  = []
    all_cond_raw      = []

    # Get normalization stats for inversion
    cond_mean = coef_norm['cond']['mean']
    cond_std  = coef_norm['cond']['std']
    out_mean  = coef_norm['output']['mean']
    out_std   = coef_norm['output']['std']

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for graph in test_loader:
            graph = graph.to(device)
            # 1) Predict fields (and unnormalize)
            field_n = inr_model.modulated_forward(
                graph.input, 
                graph.cond[graph.batch]
            )                              # (N, 1), normalized
            field_n = field_n.cpu()
            # invert normalization: (x * std + mean)
            # out_mean/out_std are arrays of length=2, but your output_dim=1,
            # so we take the first element of each
            field_unnorm = field_n * out_std + out_mean
            all_field_preds.append(field_unnorm.squeeze(-1).numpy())

            # 2) Predict scalars (and unnormalize)
            scal_n = inr_model.predict_scalars(graph.cond)  # (1, 6)
            scal_n = scal_n.cpu().squeeze(0)
            scal_unnorm = scal_n * coef_norm['scalars']['std'] + coef_norm['scalars']['mean']
            all_scalar_preds.append(scal_unnorm.numpy())

            # 3) Recover raw conditions
            cond_n = graph.cond.cpu().squeeze(0).numpy()     # (cond_dim,)
            cond_unnorm = cond_n * cond_std + cond_mean     # invert standardization
            all_cond_raw.append(cond_unnorm)

    # —————————— SAVE EVERYTHING ——————————
    results = {
        # lists of NumPy arrays, one entry per test sample
        'fields_pred':  all_field_preds,  
        'scalars_pred': all_scalar_preds,
        'conds_raw':    all_cond_raw,
        # metadata
        'cond_keys':    ['<latent_0>', '…', 'angle_in', 'angle_out'],
        'scalar_keys':  ['Pr', 'Q', 'Tr', 'angle_out', 'eth_is', 'power'],
    }

    save_path = os.path.join(results_directory, f"{run_name}_predictions.pt")
    torch.save(results, save_path)
    print(f"Saved predictions to {save_path}")
    return


if __name__ == "__main__":
    main()