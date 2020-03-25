# LinearizedNNs

Here I apply Neural Tangent Kernel, a new approach to train neural networks.

The repository is inspired by the following paper: [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient
Descent.](https://arxiv.org/abs/1902.06720)

## Files

**src/pytorch_impl/nns**        - neural networks in pytorch

**src/pytorch_impl/estimators** - estimators that can be trained:
* `SgdEstimator` - estimator that trains a normal neural network vis SGD
* `LinearizedSgdEstimator`        - estimator that trains a linearized neural network via SGD
* `MatrixExpEstimator`            - estimator that trains a linearized neural network via closed form solution, which turns out to be matrix exponentital

**src/pytorch_impl/classifier_training** - training procedure that trains an estimator and saves metrics on its way 

**notebooks** - jupyter notebooks 

## Legacy Files

**notebooks/legacy/Wide_NNs_pytorch** - my implementation of the training method from the paper.


**notebooks/leagcy/Resnet18** - my attempt, yet unsuccessful, to achieve 94% accuracy on Cifar-10 via normal ResNet.

