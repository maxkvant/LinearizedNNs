# LinearizedNNs

Here I apply Neural Kernels, a new approach to train neural networks.

The repository is inspired by the following papers:
1. [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient
Descent](https://arxiv.org/abs/1902.06720)
2. [Neural Kernels without Tangents](https://arxiv.org/abs/2003.02237)

## Files

**src/pytorch_impl/nns**        - neural networks in pytorch

**src/pytorch_impl/estimators** - estimators that can be trained:
* `SgdEstimator` - estimator that trains a normal neural network via SGD
* `LinearizedSgdEstimator`        - estimator that trains a linearized neural network via SGD
* `MatrixExpEstimator`            - estimator that trains a linearized neural network via closed form solution, which turns out to be matrix exponentital

**src/pytorch_impl/classifier_training** - training procedure that trains an estimator and saves metrics on its way 

**notebooks** - jupyter notebooks 

**papers** - PDFs of papers I might refer in future

## Legacy Files

**notebooks/legacy/Wide_NNs_pytorch** - my implementation of the training method from the paper.


**notebooks/leagcy/Resnet18** - my attempt, yet unsuccessful, to achieve 94% accuracy on Cifar-10 via ResNet18.

