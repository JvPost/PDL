# PDL2022 Assignment 3

In these files you can find the solutions for question one and two from the third PDL assignment.

## How to use this code:

There are three run-able files: `assignment3_meanfield.py`, `assignment3_covariance.py`, `plot_reconstruction.py`

Running `assignment3_meanfield.py` & `assignment3_covariance.py` will train the models for question 1 & 2 respectively on MNIST and FashionMNIST. The trained model's are saved via `torch.save` along with their training curves.
`plot_reconstructions.py` will load the trained models and pass a random examples from the test-datasets through the trained VAE. The resulting reconstruction will be plotted next to the original.