# Emergence of Latent Binary Encoding in Deep Neural Network Classifiers
Code to reproduce results presented in the work [ Emergence of Latent Binary Encoding in Deep Neural Network Classifiers](https://arxiv.org/abs/2310.08224) as part of the "Symmetry and Geometry in
Neural Representations" workshop taking place at NeurIPS 2023.

# Abstract
We observe the emergence of binary encoding within the latent space of deep-neural-network classifiers.
Such binary encoding is induced by introducing a linear penultimate layer,  which is equipped during training with a loss function that grows as $\exp(\vec{x}^2)$, where $\vec{x}$ are the coordinates in the latent space. 
The phenomenon we describe represents a specific instance of a well-documented occurrence known as neural collapse, which arises in the terminal phase of training and entails the collapse of latent class means to the vertices of a simplex equiangular tight frame (ETF).
We show that binary encoding accelerates convergence toward the simplex ETF and enhances classification accuracy.

# Run the code
The following piece of code reproduces results presented in the manuscript:

```
python main.py --config config/mnist.yml --results-dir ./results_mnist
python main.py --config config/fashion.yml --reuslts-dir ./results_fashion
```
Metrics collected during trainings on the MNIST and FashinMNIST are saved respectively in the './results_mnist' and './results_fashion' directories.
