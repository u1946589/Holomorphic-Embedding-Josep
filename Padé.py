import numpy as np
from numba import jit

#@jit(nopython=True)
def Padé_func(U, L):  # only for diagonal approximants
    """

    :param U: vector with voltage coefficients
    :param L: order of the numerator, same for the denominator
    :return: final voltage value
    """

    def conv(U, vec_q2, i):
        suma = [U[k] * vec_q2[i - k] for k in range(i + 1)]
        return sum(suma)

    RHS = np.zeros(L, dtype=complex)
    vec_q2 = np.zeros(L + 1, dtype=complex)

    RHS[:L] = -U[L + 1: L + 1 + L]
    mat = [[U[L+i-j] for j in range(L)] for i in range(L)]

    vec_q = np.linalg.solve(mat, RHS)
    vec_q2[0] = 1
    vec_q2[1:L+1] = vec_q[0:L]

    vec_p2 = [conv(U, vec_q2, i) for i in range(L + 1)]

    return sum(vec_p2)/sum(vec_q2)

