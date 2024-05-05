# needle - (necessary elements of deep learning) library

This is a implementation of mini-pytorch of [10714 deep learning sytems](https://dlsyscourse.org/) that 
- build a basic automatic differentiation framework with forward/backward passes
- build a computational graph with topological sort
- Reverse mode differentiation
- Implementing a neural network library (linear layers, ReLU, Sequential, LogSumExp, LayerNorm1d, BatchNorm1d, Dropout, Residual).
- Optimizers: (SGD and Adam) w/ momentum and weight decay 
- Dataset and Dataloader
- And finally a ResNet implementation using all things implemented above

# Packages required for running tests
```
pip install numdifftools
pip install git+https://github.com/dlsys10714/mugrade.git
```