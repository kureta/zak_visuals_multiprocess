from time import time

import numpy as np
from scipy.linalg import fractional_matrix_power, qr


def random_orthonormal(n=512):
    H = np.random.randn(n, n).astype('float32')
    Q, R = qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))

    return Q


if __name__ == '__main__':
    d = 512
    basis = np.eye(d)
    det = -1
    rotation = None
    while det < 0:
        rotation = random_orthonormal(d)
        det = np.linalg.det(rotation)

    print(rotation.dot(rotation.T))
    print(det)
    t0 = time()
    rotation = fractional_matrix_power(rotation, 1/16).real
    print(time() - t0)
    rotation = rotation / np.linalg.norm(rotation, ord=2, axis=1)
    det = np.linalg.det(rotation)
    print(rotation.dot(rotation.T))
    print(det)

    dot = np.tensordot(rotation, basis)
    angle = np.arccos(dot/d)
    print(angle)
