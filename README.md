# Vasireddy2025
Data and code for studying single-neuron responses to multi-electrode extracellular stimulation.

# Installation
First, clone the respository and create or update an existing conda environment using spec-file.txt -- this contains the requried dependencies. This can be done using

`conda create --name myenv --file spec-file.txt` or `conda install --name myenv --file spec-file.txt`

Primarily, the code requires:

- scikit-learn
- numpy
- statsmodels
- scipy
- matplotlib
- PyTorch
- GPyTorch

Next, after creating/updating the conda environment, activate the environment and navigate to the top-level directory of the repository and run

`pip install -e .`

within the conda environment. This will install the multi-site model fitting utilities as src.fitting.
