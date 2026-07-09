# Environment Setup

## 1. Install Anaconda

Source: https://www.anaconda.com/docs/getting-started/installation

## 2. Create a virtual environment

```bash
conda create -n my_env_name python=3.12 -y
```

## 3. Install dependencies

```bash
# PennyLane for quantum machine learning
pip install pennylane

# PyTorch with CUDA 12.6 for Nvidia GTX 1060 GPU
# Source: https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Jupyter lab and notebook
pip install jupyter

# Others
pip install numpy pandas scikit-learn matplotlib scipy pyarrow fastparquet
```

**Note**: [reproducible environment file](conda_environment.yaml)

## 4. Activate the virtual environment

```bash
conda activate my_env_name
```

## 5. Develop

```bash
# Open Jupyter Lab
jupyter-lab
```