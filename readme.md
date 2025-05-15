# MARIO: Multiscale Aerodynamic Resolution Invariant Operator

This repository contains the implementation of MARIO, a conditional neural field architecture for predicting aerodynamic fields around airfoils. 
MARIO achieved the 3rd Place at the [ML4CFD Challenge at Neurips 2024](https://neurips.cc/virtual/2024/competition/84799) and is currently one of the top models on the [PAID benchmark](https://huggingface.co/PLAIDcompetitions) on Hugging Face.

MARIO is a conditional Neural Field architecture designed for surrogate modeling of large PDEs and it is specialized for aerodynamic applications, although the framework is absolutely general. It can handle paramteric and non parametric geometric variability. In the latter case, a Neural Field encoder is used to learn a latent code on the Signed Distance Function fields (as in the Airfrans and VKI-BLADE test cases). You can refer to the original paper for more information:
![MARIO Architecture](figures/new_overview_final.png)
*MARIO's architecture overview: A conditional neural field with multiscale Fourier features and hypernetwork modulation.*


## 📁 Repository Structure
- **airfrans_task/**  : MARIO code for the  [Airfrans dataset](https://airfrans.readthedocs.io/en/latest/index.html). Employes geometry encoding on the Signed Distance Function: first the shape encoding are learned using a Neural Field encoder (by running train_sdf.py). Then the output model (MARIO) is used to learn to map the coordinates, the inflow conditions and the learnt geometric latent codes to the output fields. Different splits can be used by specifyng the task in the configurations file.
- **vki_task/**       : MARIO code for the [VKI-LS59 Dataset](https://huggingface.co/spaces/PLAIDcompetitions/VKILS59Benchmark). Here the two stage training process is used: first the shape encoding are learned using a Neural Field encoder (by running train_sdf.py). Then the output model (MARIO) is used to learn to map the coordinates, the inflow conditions and the learnt geometric latent codes to the output fields and scalars.
- **airfrans_task_challenge/**  : Original MARIO code for the  [Airfrans dataset](https://airfrans.readthedocs.io/en/latest/index.html) used in the Neurips Challenge The data handling relies on the library  [pyoche](https://pypi.org/project/pyoche/). The code is structured with callbacks. A simplified geometry encoding process is used, by using the camber and thickness distribution. More info can be found inside the folder.

## Requirements
The Airfrans dataset can be obtained by installing the airfrans package. The data handling is done with the custom library pyoche.
```bash
pip install airfrans

```

#### 📥 Data

##### Airfrans

The original Airfrans dataset is provided via the `airfrans` Python package.

1. **Install**

   ```bash
   pip install airfrans
   ```

##### VKI Blade (VKILS59)

More info on the VKILS59 benchmark:
[https://huggingface.co/spaces/PLAIDcompetitions/VKILS59Benchmark](https://huggingface.co/spaces/PLAIDcompetitions/VKILS59Benchmark)

---
## Usage

Look into each specific folder.
