import numpy as np

"""
Based on part of the following file https://github.com/modestyachts/neural_kernels_code/blob/master/utils.py

"""


def to_zca(train, test, zca_bias=0.0001):
    orig_train_shape = train.shape
    orig_test_shape = test.shape

    train = np.ascontiguousarray(train, dtype=np.float32).reshape(train.shape[0], -1).astype(np.float64)
    test = np.ascontiguousarray(test, dtype=np.float32).reshape(test.shape[0], -1).astype(np.float64)

    n_train = train.shape[0]

    # Zero mean every feature
    train = train - np.mean(train, axis=1)[:,np.newaxis]
    test = test - np.mean(test, axis=1)[:,np.newaxis]

    # Normalize
    train_norms = np.linalg.norm(train, axis=1)
    test_norms = np.linalg.norm(test, axis=1)

    # Make features unit norm
    train = train/train_norms[:,np.newaxis]
    test = test/test_norms[:,np.newaxis]

    train_cov_mat = 1.0/n_train * train.T.dot(train)

    (E,V) = np.linalg.eig(train_cov_mat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E)
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)

    train = (train).dot(global_ZCA)
    test = (test).dot(global_ZCA)

    return (train.reshape(orig_train_shape).astype(np.float64), test.reshape(orig_test_shape).astype(np.float64)), global_ZCA
