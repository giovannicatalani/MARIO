import os
import yaml
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
import sys
sys.path.append(str(Path(__file__).parents[1]))
import airfrans as af
from dataset import AirfransFlowDataset
from src.load_models import load_inr


def evaluate(run_dir, device="cuda"):
    """
    Evaluate a pretrained INR model on both train and test datasets.

    Args:
        run_dir (str): Path to training run folder (must contain config.yaml and model checkpoint).
        device (str): "cuda" or "cpu"
    """

    # 1) Load config
    cfg_path = os.path.join(run_dir, "config.yaml")
    with open(cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)

    # 2) Load checkpoint
    ckpt_files = [f for f in os.listdir(run_dir) if f.endswith(".pt")]
    if not ckpt_files:
        raise FileNotFoundError("No .pt checkpoint found in run directory")
    ckpt_path = os.path.join(run_dir, ckpt_files[0])
    checkpoint = torch.load(ckpt_path, map_location=device)
    print(f"Loaded checkpoint from {ckpt_path}")

    # 3) Reload datasets
    train_data, train_names = af.dataset.load(root=cfg.dataset.root_path, task=cfg.dataset.task, train=True)
    test_data, test_names   = af.dataset.load(root=cfg.dataset.root_path, task=cfg.dataset.task, train=False)

    train_latents = np.load(cfg.dataset.train_latents_path)
    test_latents  = np.load(cfg.dataset.test_latents_path)

    # Training dataset (to recompute coef_norm)
    train_ds = AirfransFlowDataset(
        train_data,
        train_names,
        geom_latents=train_latents["train_modulations"],
        mode="train",
    )
    coef_norm = train_ds.coef_norm

    # Test dataset (reuse train stats)
    test_ds = AirfransFlowDataset(
        test_data,
        test_names,
        geom_latents=test_latents["val_modulations"],
        mode="test",
        coef_norm=coef_norm,
    )

    # 4) Rebuild model and load weights
    in_dim = cfg.inr.in_dim
    out_dim = cfg.inr.out_dim
    inr_model = load_inr(checkpoint["inr_in"], cfg, in_dim, out_dim, device=device)
    inr_model.eval()

    # Normalization parameters
    out_mean = coef_norm["output"]["mean"]
    out_std  = coef_norm["output"]["std"]
    output_fields = ["Velocity-x", "Velocity-y", "Pressure", "Turbulent-viscosity"]
    out_mean_array = np.array([out_mean[f] for f in output_fields])
    out_std_array  = np.array([out_std[f] for f in output_fields])

    # 5) Predict helper
    def predict_dataset(dataset, split_name):
        preds = []
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for graph in loader:
                graph = graph.to(device)
                field_n = inr_model.modulated_forward(
                    graph.input,
                    graph.cond[graph.batch],
                )
                field_n = field_n.cpu().numpy()
                field_unnorm = field_n * out_std_array + out_mean_array
                preds.append(field_unnorm)
        return preds

    print("Predicting on training set...")
    fields_train = predict_dataset(train_ds, "train")

    print("Predicting on test set...")
    fields_test = predict_dataset(test_ds, "test")

    # 6) Save results
    results = {
        "fields_pred_train": fields_train,
        "fields_pred_test": fields_test,
    }
    save_path = os.path.join('/scratch/dmsm/gi.catalani/Airfrans/', "predictions.pt")
    torch.save(results, save_path)
    print(f"Saved predictions to {save_path}")


if __name__ == "__main__":
    # Example usage: python evaluate.py /path/to/run_dir
    import sys
    run_dir = '/home/dmsm/gi.catalani/Projects/mario_challenge/airfrans_task/trainings/training_result_20250926-140207/'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(run_dir, device)
