# SSSR

This repository provides the implementation and numerical examples for the shared-structure sparse regression (SSSR) framework.

## Requirements

### Hardware requirements

All tests were performed on a workstation with:

- AMD Ryzen 7 7800X3D 8-Core CPU,
- 32.0 GB RAM, and
- NVIDIA GeForce RTX 4070 Ti GPU.

### Software requirements

We strongly recommend using Anaconda for environment management and running the examples with Jupyter Notebook. The main software and packages used in this work are:

- NVIDIA GPU driver
- CUDA Toolkit
- Anaconda
- Jupyter Notebook
- Python 3.11
- PyTorch 2.2.2
- NumPy
- SciPy
- Matplotlib
- meshio

## Running the examples

To run the examples, open the corresponding Jupyter notebook files in each example directory. The examples include:

- 1D Burgers equation
- 2D Burgers equations
- reaction--diffusion system
- Navier--Stokes equations

The datasets are included in this repository, except for the Navier--Stokes case, whose dataset exceeds 5 GB.
