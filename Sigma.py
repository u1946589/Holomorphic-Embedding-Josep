import numpy as np


def Sigma_func(U, L):
    """

    :param U: vector with voltage coefficients
    :param L: order of the numerator, same for the denominator
    :return: sigma complex value
    """

    cs = np.zeros(len(U)-1, dtype=complex)
    cs[:len(U)-1] = U[1:len(U)]

    ds = np.zeros(len(U), dtype=complex)
    ds[0] = 1/np.conj(U[0])

    def sumds(ds,U,i):
        suma = [ds[k]*np.conj(U[i-k]) for k in range(i)]
        return sum(suma)

    ds[1:len(U)] = [-sumds(ds, U, i) / np.conj(U[0]) for i in range(1, len(U))]

    mat1 = [[cs[i - j - 1] if i > j else 0 for j in range(L)] for i in range(2 * L + 1)]
    mat2 = [[-ds[i - j] if i >= j else 0 for j in range(L + 1)] for i in range(2 * L + 1)]
    mat = [[mat1[i][j] if j < L else mat2[i][j-L] for j in range(2*L+1)] for i in range(2*L+1)]

    LHS = np.dot(np.linalg.inv(mat), -cs)

    q = sum(LHS[:L])
    p = sum(LHS[L:])

    return (p/(q+1)) #+1 because q0 = 1
