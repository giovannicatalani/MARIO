# Rotor37 Task

This folder contains the training pipeline for the **Rotor37**  dataset using the MARIO architecture.

---

## Preprocessing
As the Signed Distance Function Fields are not directly available in the Plaid dataset, a preprocessing script: `preprocess_sdf.py` is provided.
This script computes the signed distance function around each 3D Blade shapes, and saves the complete dataset (including the other fields) in format .h5 using the pyoche library. It is possible to save in different formats just by modifying the script. The output folders, and input folders need to be specified accordingly.

## Workflow Overview
Once the pyoche dataset is ready after the preprocessing, MARIO can be trained following the usual pipeline: 

1. **SDF Encoder Training** (`train_sdf.py`)

   * Configured via `config_sdf.yaml`
   * Produces latent **modulations** for each blade geometry.
   * Modulations are saved under the `trainings/training_sdf_<timestamp>/modulations/` directory.

2. **Flow Model Training** (`train_flow.py`)

   * Uses the precomputed SDF latents as geometric conditioning.
   * Specify the path to the saved `.npz` modulations in `config_out.yaml` under `dataset.train_latents_path` and  `dataset.test_latents_path`.

---

## Usage

### 1. Train SDF Encoder

```bash
cd 2Dprofile_task
python train_sdf.py \
```

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

Data is retrieved from [https://huggingface.co/spaces/PLAIDcompetitions/Rotor37Benchmark](https://huggingface.co/spaces/PLAIDcompetitions/Rotor37Benchmark), more info can be found there.
The library plaid must be installed to load the data.
