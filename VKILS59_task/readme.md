# VKI Blade Task

This folder contains the training pipeline for the **VKILS59** blade dataset using the MARIO architecture.

---

![MARIO VKI](figures/mario_vki_overview-1.png)
![MARIO Encoding](figures/gemetry_encoding_mario_vki-1.png)


## Workflow Overview

1. **SDF Encoder Training** (`train_sdf.py`)

   * Configured via `config_sdf.yaml`
   * Produces latent **modulations** for each blade geometry.
   * Modulations are saved under the `trainings/training_sdf_<timestamp>/modulations/` directory.

2. **Flow Model Training** (`train_flow.py`)

   * Uses the precomputed SDF modulations as geometric conditioning.
   * Specify the path to the saved `.npz` modulations in `config_out.yaml` under `dataset.train_latents_path` and  `dataset.test_latents_path`.

---

## Usage

### 1. Train SDF Encoder

```bash
cd VKILS59_task
python train_sdf.py \
```

* **Outputs:**

  * `outputs/sdf_encoder_<timestamp>/modulations/train_<latent_dim>.npz`
  * `outputs/sdf_encoder_<timestamp>/modulations/test_<latent_dim>.npz`

### 2. Train Flow Model

Edit `config_out.yaml` to set modulations path.

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
python train_flow.py optim.lr_flow=5e-4
```

---

## Data

Before running, ensure you have converted the raw Plaid-format VKILS59 data into Pyoche `.pch` files:

```bash
python convert_to_pch.py \
  problem.path=raw/problem_definition \
  dataset.path=raw/dataset \
  out_dir=pyoche_data
```
This is stil under progress.
Load these in your config under:

```yaml
dataset:
  train_pch: vki_task/pyoche_data/train.pch
  test_pch:  vki_task/pyoche_data/test.pch
```

*Data handling is done via the `pyoche` library.*
