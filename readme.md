# MARIO: Multiscale Aerodynamic Resolution Invariant Operator

This repository contains the implementation of MARIO, a conditional neural field architecture for predicting aerodynamic fields around airfoils. 
MARIO achieved the 3rd Place at the ML4CFD Challenge at Neurips 2024.

![MARIO Architecture](figures/new_overview_final.png)
*MARIO's architecture overview: A conditional neural field with multiscale Fourier features and hypernetwork modulation.*


## 📁 Repository Structure

- **airfrans_task/**  : Original MARIO code for the Airfrans dataset.  
- **vki_task/**       : New code for the VKILS59 blade dataset (SDF + flow).  


## Requirements
The Airfrans dataset can be obtained by installing the airfrans package. The data handling is done with the custom library pyoche.
```bash
pip install airfrans
pip install pyoche
```

#### 📥 Data

##### Airfrans

The original Airfrans dataset is provided via the `airfrans` Python package.

1. **Install**

   ```bash
   pip install airfrans
   ```
2. **Download & prepare**
   Convert the raw Airfrans files into Pyoche format:

   ```bash
   cd airfrans_task
   python save_to_pch.py dataset.path=/path/to/airfrans/data
   ```
3. **Output**

   * `airfrans_task/pyoche_data/train.pch`
   * `airfrans_task/pyoche_data/test.pch`


##### VKI Blade (VKILS59)

More info on the VKILS59 benchmark:
[https://huggingface.co/spaces/PLAIDcompetitions/VKILS59Benchmark](https://huggingface.co/spaces/PLAIDcompetitions/VKILS59Benchmark)

---
## Usage

Look into each specific folder.
