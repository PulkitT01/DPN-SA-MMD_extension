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

