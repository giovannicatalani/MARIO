import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

i
import hydra
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

import torch.optim as optim
import random
import matplotlib.pyplot as plt
# allow imports of your project
import sys
sys.path.append(str(Path(__file__).parents[1]))

from utils.dataset import Hyper_SDFDataset, subsample_dataset
from utils.utils_training import graph_outer_step
from utils.load_models import create_inr_instance, load_inr

from pre_dataset import pre_process_dataset


@hydra.main(config_path=".", config_name="config_sdf.yaml")
def main(cfg: DictConfig) -> None:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_dtype(torch.float32)

    # ───── Create timestamped results directory ─────────────────────────────────
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base = Path(cfg.output_dir)
    results_dir = base / 'trainings' / f"training_sdf_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # save config for reproducibility
    with open(results_dir / 'config.yaml', 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

    # —── Data loading

    train_raw, test_raw = pre_process_dataset()



    train_ds = Hyper_SDFDataset(train_raw, is_train=True)
    test_ds  = Hyper_SDFDataset(test_raw, is_train=False, coef_norm=train_ds.coef_norm)

    n_train = len(train_ds)
    n_test  = len(test_ds)

    print(f"n_train = {n_train}")
    print(f"n_test = {n_test}")


    # —── Build INR model
    inr_model = create_inr_instance(
        cfg,
        input_dim  = cfg.inr.input_dim,
        output_dim = cfg.inr.output_dim,
        device     = device
    )
    inr_model.to(device)
  




    # —── Optimizer & hyper‐lr for alpha
    alpha_in = nn.Parameter(torch.Tensor([cfg.optim.lr_code]).to(device))
    optimizer = torch.optim.AdamW([
        {"params": inr_model.parameters()},
        {"params": alpha_in, "lr": cfg.optim.meta_lr_code, "weight_decay": 0},
    ], lr=cfg.optim.lr_inr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.99,
        patience=200, min_lr=1e-5
    )
    train_loss_history = []
    test_loss_history = []

    # —── Training loop
    best_val = float('inf')
    for epoch in tqdm(range(cfg.optim.epochs), desc="Epoch"):

        print(epoch)
        # --- TRAIN ---
        inr_model.train()
        train_loss = 0.0





        train_loader = DataLoader(
            subsample_dataset(train_ds, cfg.optim.num_points, mode='1'),
            batch_size=cfg.optim.batch_size,
            shuffle=True
        )
    
        for batch in train_loader:
            batch = batch.to(device)
            batch.modulations = torch.zeros(
                cfg.optim.batch_size, cfg.inr.latent_dim, device=device
            )
         
            out = graph_outer_step(
                inr_model, batch,
                cfg.optim.inner_steps,
                alpha_in,
                epoch ,
                num_batch = compteur,
                is_train=True,
            )
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(inr_model.parameters(), clip_value=1.0)
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs

        avg_train = train_loss / n_train
        train_loss_history.append({'epoch': epoch, 'loss': avg_train})


        print(f"Loss/Train = {avg_train} , Epoch = {epoch}")

        # --- VALIDATION every 5 epochs ---
        if epoch % 5 == 0:
            inr_model.train()
            val_loss = 0.0

            val_loader = DataLoader(
                subsample_dataset(test_ds, cfg.optim.num_points),
                batch_size=cfg.optim.batch_size, shuffle=False
            )

            for batch in val_loader:
                batch = batch.to(device)
                batch.modulations = torch.zeros(
                    cfg.optim.batch_size, cfg.inr.latent_dim, device=device
                )
                out = graph_outer_step(
                    inr_model, batch,
                    cfg.optim.inner_steps,
                    alpha_in,
                    num_epoch = 100000,
                    num_batch = 80,
                    is_train=False,

                )
                val_loss += out["loss"].item() * batch.num_graphs

            avg_val = val_loss / n_test

            print(f"Loss/Val = {avg_val} , Epoch = {epoch}")
            scheduler.step(avg_val)



            test_loss_history.append({'epoch': epoch, 'loss': avg_val}) #'field_loss': test_loss_field})#, 'scalar_loss': test_loss_scalar})


            plt.figure()
            plt.plot([p['epoch'] for p in train_loss_history], [p['loss'] for p in train_loss_history], label='Train Loss')
            plt.plot([p['epoch'] for p in test_loss_history], [p['loss'] for p in test_loss_history], label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.title('Training and Test Loss')
            plt.legend()
            plt.savefig("PLOT_SDF_loss.png")
            plt.close()

            # save best model checkpoint
            if avg_val < best_val:
                best_val = avg_val
                torch.save(
                    {
                        "epoch": epoch,
                        "inr_in": inr_model.state_dict(),
                        "alpha_in": alpha_in.detach().cpu(),
                        "optimizer": optimizer.state_dict(),
                    },
                    results_dir / "best_inr.pt"
                )

    # —── Extract & Save Modulations from best model ─────────────────────────
    ckpt = torch.load(results_dir / "best_inr.pt", map_location=device)
    inr_model = load_inr(ckpt["inr_in"], cfg, cfg.inr.input_dim, cfg.inr.output_dim, device=device)
    inr_model.eval()

    # train modulations
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    train_mods = []

    for batch in train_loader:
        batch = batch.to(device)
        batch.modulations = torch.zeros(batch.num_graphs, cfg.inr.latent_dim, device=device)
        out = graph_outer_step(inr_model, batch, cfg.optim.inner_steps, ckpt["alpha_in"].to(device),0,0, is_train=False)
        train_mods.append(out["modulations"].detach().cpu().numpy())
    train_mods = np.concatenate(train_mods, axis=0)




    # test modulations
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    test_mods = []

    for batch in test_loader:
        batch = batch.to(device)
        batch.modulations = torch.zeros(batch.num_graphs, cfg.inr.latent_dim, device=device)
        out = graph_outer_step(inr_model, batch, cfg.optim.inner_steps, ckpt["alpha_in"].to(device),0,0, is_train=False)
        test_mods.append(out["modulations"].detach().cpu().numpy())
    test_mods = np.concatenate(test_mods, axis=0)

    mod_dir = results_dir / 'modulations'
    mod_dir.mkdir(exist_ok=True)
    np.savez(mod_dir / f"{cfg.dataset.task}_train_{cfg.inr.latent_dim}.npz",
             train_modulations=train_mods)
    np.savez(mod_dir / f"{cfg.dataset.task}_test_{cfg.inr.latent_dim}.npz",
             val_modulations=test_mods)
    print(f"Saved modulations: train {train_mods.shape}, test {test_mods.shape}")






if __name__ == "__main__":
    main()
