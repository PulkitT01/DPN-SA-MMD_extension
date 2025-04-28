# Deep Propensity Network using Sparse Autoencoder (DPN-SA) with MMD Extension

## Overview
This repository implements **DPN-IPM**, an extension of the **Deep Propensity Network using a Sparse Autoencoder (DPN-SA)**, designed for robust estimation of treatment effects from high-dimensional observational data. While DPN-SA uses a KL-divergence-based regularization to enforce sparsity in latent space, **DPN-IPM replaces this with a Maximum Mean Discrepancy (MMD)** penalty, aligning latent distributions more effectively using an IPM-based regularization approach.

## Key Contributions
- Replacement of KL-based sparsity regularizer with an **MMD-based regularization** (Integral Probability Metric)
- Distribution-level alignment of latent codes for better **covariate balancing**
- Three training modes: **End-to-End**, **Stacked-All**, and **Stacked-Cur**
- Evaluation using **Jobs dataset**, with focus on ATE, ATT, and policy risk
- Empirical gains in stability and accuracy of causal effect estimates

## Features
- Modular architecture with Sparse Autoencoder for propensity score estimation
- MMD-based regularization in latent space using **RBF kernel**
- Deep Counterfactual Network (DCN) for potential outcome prediction
- End-to-end and layer-wise training configurations
- Evaluation of models on multiple causal metrics: **ATE**, **ATT**, **Policy Risk**

## Installation
### Prerequisites
- Python 3.10.12
- torch==2.5.1+cu121
- numpy==2.1.2
- pandas==2.2.3
- matplotlib==3.10.0
- scikit-learn==1.6.0

## Usage
### Running Experiments
To train and evaluate DPN-IPM or DPN-SA models on the Jobs dataset:
```bash
python main_propensity_dropout.py
```

### Implementing MMD Regularization
MMD-based regularization is integrated into the Sparse Autoencoder in `DPN_SA_Deep.py` and `Utils.py`. The core logic uses RBF kernel to compute the MMD between latent activations and a Bernoulli reference distribution.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## References
- Ghosh et al. (2021). *Deep Propensity Network using a Sparse Autoencoder*. Journal of the American Medical Informatics Association.
- Tolstikhin et al. (2018). *Wasserstein Autoencoders*. ICLR.
- [Original DPN-SA GitHub](https://github.com/shantanu-ai/DPN-SA)

## Acknowledgements
- This repository builds on the original DPN-SA implementation.
- The **MMD loss** function is adapted from the [Kaggle notebook by Onur Tunali](https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy).
- Supervised by Roberto Faleh & Sofia Morelli at Eberhard Karls Universität Tübingen.

## IHDP Dataset Integration
To extend the architecture to support the IHDP dataset (continuous outcomes), the following modifications were made across key scripts:

| Component | Changes | Reasoning |
|:---|:---|:---|
| **DCN.py** | Added `output_dim` in `__init__()`; made output layer dynamic. | To flexibly support 2-class (Jobs) and real-valued (IHDP) outputs. |
| **DCN_network.py** | Introduced `running_mode` to dynamically adjust:<br>- Loss function (CrossEntropy vs MSE)<br>- Output format (argmax vs raw output)<br>- Output dimension (2 vs 1) | Needed to handle classification (Jobs) vs regression (IHDP) seamlessly. |
| **DPN_SA_Deep.py** | Added `running_mode` in training and testing methods, passed it through DCN phases. | Ensured mode consistency across all training and evaluation steps. |
| **Experiments.py** | Added "ihdp" handling:<br>- Dataset paths<br>- `input_nodes=25`<br>- Output locations. | Enabled running experiments on Jobs, IHDP, and synthetic datasets without code rewrites. |
| **mmd_DPN_SA_Deep.py** | Same as DPN_SA_Deep.py: running_mode added everywhere for MMD-regularized version. | Correct mode-specific behavior while using MMD in SAE models. |
| **mmd_Experiments.py** | Added "ihdp" support:<br>- Paths, parameters updated.<br>- Focused only on SAE models (NN/LR dropped for MMD experiments). | Needed a clean pipeline for IHDP with MMD extension. |
| **Graphs.py** | Changed:<br>- Dataset loaded to IHDP.<br>- `input_nodes=25`.<br>- Titles updated to "ihdp".<br>- Model training and plot logic remains same.<br>- **Input nodes hardcoded to 25 for IHDP**. | Adapted graphs for IHDP without changing plotting functions. |

### Notes
- **Graphs.py**: For IHDP, input feature size is **hardcoded** to **25**. (Jobs version had it hardcoded to 17.)
- Dataset paths and model save paths are statically set based on `running_mode` ("jobs", "ihdp", "synthetic_data").
- Only SAE-based architectures are evaluated under MMD regularization.
