import setuptools


setuptools.setup(
    name='linearized_nns',
    version='0.0.1',
    package_dir={'': 'src'},
    packages=['linearized_nns',
              'linearized_nns/from_neural_kernels',
              'linearized_nns/pytorch_impl',
              'linearized_nns/pytorch_impl/estimators',
              'linearized_nns/pytorch_impl/nns']
)