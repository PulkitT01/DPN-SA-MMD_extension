# Deep Propensity Network using Sparse Autoencoder (DPN-SA) with MMD Extension

## Overview
This repository contains an implementation of the **Deep Propensity Network using a Sparse Autoencoder (DPN-SA)** for estimating treatment effects from observational data. The original DPN-SA model was proposed to address high-dimensional propensity score matching and counterfactual outcome prediction, utilizing deep learning approaches.

This repository extends the original implementation by incorporating **Maximum Mean Discrepancy (MMD)** as an alternative to the **b-KL loss**, improving the robustness of treatment effect estimation.

## Features
- Implementation of **DPN-SA** for causal inference
- Sparse Autoencoder-based feature extraction for propensity score estimation
- Counterfactual outcome prediction using deep learning
- **MMD-based loss implementation** replacing b-KL loss
- Support for randomized and real-world datasets
- Performance evaluation with various models (Logistic Regression, LASSO, DCN, DPN-SA)

## Installation
### Prerequisites
- Python 3.10.12
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Setup
1. Clone this repository:
   ```sh
   git clone https://github.com/PulkitT01/DPN-SA-MMD_extension.git
   cd DPN-SA-MMD_extension
   ```
2. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows
   ```
3. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Running Experiments
To run the training and evaluation pipeline:
```sh
python main_propensity_dropout.py
```
This script will train and evaluate DPN-SA models on provided datasets.

### Implementing MMD Loss
To modify the loss function to use **Maximum Mean Discrepancy (MMD)** instead of **b-KL**, update the relevant sections in `DPN_SA_Deep.py`.

## Repository Structure
```
DPN-SA-MMD_extension/
│── Dataset/                     # Contains dataset files
│   ├── columns.txt
│   ├── ihdp_sample.csv
│   ├── jobs_DW_bin.new.10.train.npz
│   ├── jobs_DW_bin.new.10.test.npz
│
│── DCN/                         # Deep Counterfactual Networks (DCN) implementation
│   ├── DCN_network.py
│   ├── DCN.py
│   ├── Model25_10_25.py
│
│── DPN_SA/                      # Deep Propensity Network (DPN-SA) implementation
│   ├── DPN_SA_Deep.py
│   ├── Experiments.py
│   ├── pm_match.py
│
│── Graphs/                      # Visualization and graphing utilities
│   ├── Graphs.py
│
│── PropensityModels/             # Propensity score estimation models
│   ├── Propensity_net_NN.py
│   ├── Propensity_score_LR.py
│   ├── Sparse_Propensity_net.py
│   ├── Sparse_Propensity_net_shallow.py
│   ├── Sparse_Propensity_score.py
│   ├── shallow_net.py
│   ├── shallow_train.py
│
│── Results/                      # Experiment outputs
│   ├── Logs/                     # Logs of model runs
│   ├── Models/                   # Saved model checkpoints
│   ├── Output/                   # Final model outputs and performance metrics
│
│── Utils/                        # Utility scripts for data handling and preprocessing
│   ├── dataloader.py
│   ├── Utils.py
│
│── job.slurm                     # Slurm job script for cluster execution
│── main_propensity_dropout.py    # Main script for running experiments
│── LICENSE                       # License information
│── README.md                     # Documentation file
│── structure.md                   # Additional repo structure details
│── venv/                         # Virtual environment directory (optional)
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## References
- [Original DPN-SA Paper](https://pubmed.ncbi.nlm.nih.gov/33594415/)
- [GitHub Repository of Original DPN-SA](https://github.com/shantanu-ai/DPN-SA)

