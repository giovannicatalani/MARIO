# MARIO: Multiscale Aerodynamic Resolution Invariant Operator

This repository contains the implementation of MARIO, a conditional neural field architecture for predicting aerodynamic fields around airfoils. 
MARIO achieved the 3rd Place at the ML4CFD Challenge at Neurips 2024.

![MARIO Architecture](figures/mario_architecture-1.png)
*MARIO's architecture overview: A conditional neural field with multiscale Fourier features and hypernetwork modulation.*

## Requirements
The Airfrans dataset can be obtained by installing the airfrans package. The data handling is done with the custom library pyoche.
```bash
pip install airfrans
pip install pyoche
```

## Usage

### Training

The training script uses Hydra for configuration management. The main configuration parameters are defined in `config.yaml`. The training script expects to load the data in the pyoche format, to convert the Airfrans dataset in pyoche format it is possible to run the script:
```bash
# Train with default configuration
python save_to_pch.py
```
making sure to indicate the location to the airfrans dataset path on your local machine.
 To train the model:

```bash
# Train with default configuration
python train.py
```
making sure to indicate the location of the dataset (pre-saved in pyoche format) in the config.yaml file.

#### Override specific parameters

```bash
python train.py training.learning_rate=0.001 training.batch_size=4
```
