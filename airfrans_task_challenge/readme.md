# Airfrans Task

![MARIO Architecture](/figures/mario_architecture-1.png)
*MARIO’s multiscale Fourier features + hypernetwork modulation.*

---

**Geometric conditioning** currently uses:

* Camber line
* Thickness distribution

> ⚠️ SDF encoding support is not yet implemented.

---

## Usage

1. **Convert** the Airfrans dataset to Pyoche format:

   ```bash
   cd airfrans_task
   python save_to_pch.py \
     dataset.path=/path/to/airfrans/data \
     out_dir=pyoche_data
   ```
2. **Train** the MARIO model:

   ```bash
   python train.py
   ```
3. **Override** any Hydra parameter:

   ```bash
   python train.py optim.batch_size=8 inr.latent_dim=32
   ```
