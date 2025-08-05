# 2D_MultiScHypEl_task Task

This folder contains the training pipeline for the **2D_Multiscale_Hyperelasticity**  dataset using the MARIO architecture.

---




## Workflow Overview

1. **SDF Encoder Training** (`train_sdf.py`)

   * Configured via `config_sdf.yaml`
   * Produces latent **modulations** for each geometry.
   * Modulations are saved under the `trainings/training_sdf_<timestamp>/modulations/` directory.

2. **Flow Model Training** (`train.py`)

   * Uses the precomputed SDF latents as geometric conditioning.
   * Specify the path to the saved `.npz` modulations in `config_out.yaml` under `dataset.train_latents_path` and  `dataset.test_latents_path`.
   * Specify the output directory for the submission file 'reference'.




---

## Usage

### 1. Train SDF Encoder

```bash
cd 2D_MultiScHypEl_task
python train_sdf.py \
```

### 2. Train Flow Model

Edit `config_out.yaml` to set modulations path, and the output dir

Then run:

```bash
python train.py \

```

* **Outputs:**

  * Checkpoints under `trainings/training_result_<timestamp>/`
  * Loss curves and validation plots in the same directory as well as filed and scalar predictions.

### Override Parameters

You can override any Hydra parameter on the command line. For example:

```bash
# Change batch size and latent dimension
python train_sdf.py optim.batch_size=8 inr.latent_dim=32

# Change learning rate for flow model
python train.py optim.lr_flow=5e-4
```


## Dependencies

- `einops` = 0.8.1
- `hydra-core`= 1.3.2
- `muscat`= 2.4.1
- `omegaconf`
- `plaid`= 0.1.6 
- `torch` = 2.7.1
- `torch-geometric` = 2.6.1




---




