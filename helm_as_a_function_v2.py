import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, factorized

def conv(A, B, c, i, tipus):
    """
    3 different types of convolutions
    :param A: first vector
    :param B: the other vector
    :param c: current depth
    :param i: which bus we have selected
    :param tipus: kind of convolution needed
    :return: result of the convolution
    """
    if tipus == 1:
        suma = [np.conj(A[k, i]) * B[c - k, i] for k in range(1, c + 1)]
        return sum(suma)
    elif tipus == 2:
        suma = [A[k, i] * B[c - 1 - k, i] for k in range(1, c)]
        return sum(suma)
    elif tipus == 3:
        suma = [A[k, i] * np.conj(B[c - k, i]) for k in range(1, c)]
        return sum(suma)

def helm_josep(n, Y, vec_Y0, V0, S0, vec_shunts, pq, pv, vd, n_coeff=30):
    """

    :param n: number of buses
    :param Y: Admittance matrix
    :param vec_Y0: vector of series admittances of the branches connected to the slack bus (length n-1)
    :param V0: vector of set voltages (length n)
    :param S0: vector of set power injections (length n)
    :param vec_shunts: vector of nodal shunts (length n)
    :param pq: list of PQ node indices
    :param pv: list of PV bus indices
    :param vd: list of SLack bus indices
    :param n_coeff: number of coefficients
    :return: HELM voltage
    """
    pqpv = np.r_[pq, pv]
    np.sort(pqpv)

    # --------------------------- PREPARING IMPLEMENTATION
    vec_V = np.abs(V0[pqpv]) - 1.0  # data of voltage magnitude
    vec_W = vec_V * vec_V  # voltage magnitude squared

    U = np.zeros((n_coeff, n - 1), dtype=complex)  # voltages
    U_re = np.zeros((n_coeff, n - 1), dtype=complex)  # real part of voltages
    U_im = np.zeros((n_coeff, n - 1), dtype=complex)  # imaginary part of voltages
    X = np.zeros((n_coeff, n - 1), dtype=complex)  # X=1/conj(U)
    X_re = np.zeros((n_coeff, n - 1), dtype=complex)  # real part of X
    X_im = np.zeros((n_coeff, n - 1), dtype=complex)  # imaginary part of X
    Q = np.zeros((n_coeff, n - 1), dtype=complex)  # unknown reactive powers

    npq = len(pq)
    npv = len(pv)
    V_slack = V0[vd]
    G = Y.real
    B = Y.imag
    vec_P = S0.real
    vec_Q = S0.imag

    # .......................CALCULATION OF TERMS [0]
    Y = csc_matrix(Y)
    U[0, :] = spsolve(Y, vec_Y0)
    X[0, :] = 1 / np.conj(U[0, :])
    U_re[0, :] = np.real(U[0, :])
    U_im[0, :] = np.imag(U[0, :])
    X_re[0, :] = np.real(X[0, :])
    X_im[0, :] = np.imag(X[0, :])
    # .......................CALCULATION OF TERMS [0]. DONE

    # .......................CALCULATION OF TERMS [1]
    dimensions = 2 * npq + 3 * npv  # number of unknowns

    valor = np.zeros(n - 1, dtype=complex)
    valor[pq - 1] = (V_slack - 1) * vec_Y0[pq - 1, 0] + (vec_P[pq - 1, 0] - vec_Q[pq - 1, 0] * 1j) * X[0, pq - 1] + U[
        0, pq - 1] * vec_shunts[pq - 1, 0]
    valor[pv - 1] = (V_slack - 1) * vec_Y0[pv - 1, 0] + (vec_P[pv - 1, 0]) * X[0, pv - 1] + U[0, pv - 1] * vec_shunts[
        pv - 1, 0]

    RHSx = np.zeros((3, n - 1), dtype=float)
    RHSx[0, pq - 1] = valor[pq - 1].real
    RHSx[1, pq - 1] = valor[pq - 1].imag
    RHSx[2, pq - 1] = np.nan  # to later delete

    RHSx[0, pv - 1] = valor[pv - 1].real
    RHSx[1, pv - 1] = valor[pv - 1].imag
    RHSx[2, pv - 1] = vec_W[pv - 1, 0] - 1

    rhs = np.matrix.flatten(RHSx, 'f')
    rhs = rhs[~np.isnan(rhs)]  # delete the dummy cells

    mat = np.zeros((dimensions, 2 * (n - 1) + npv), dtype=complex)  # constant matrix
    k = 0  # index that will go through the rows
    for i in range(n - 1):  # fill the matrix
        lx = 0  # index that will go through the columns
        for j in range(n - 1):
            mat[k, lx] = G[i, j]
            mat[k + 1, lx] = B[i, j]
            mat[k, lx + 1] = -B[i, j]
            mat[k + 1, lx + 1] = G[i, j]
            if i + 1 in pq:
                if j + 1 in pq:
                    lx = lx + 2
                else:
                    lx = lx + 3
            else:
                if j + 1 not in pv:
                    lx = lx + 2  # 2 columns done
                else:
                    if j == i:
                        mat[k + 2, lx] = 2 * U_re[0, i]
                        mat[k + 2, lx + 1] = 2 * U_im[0, i]
                        mat[k, lx + 2] = -X_im[0, i]
                        mat[k + 1, lx + 2] = X_re[0, i]
                    lx = lx + 3
        if i + 1 in pq:
            k = k + 2
        else:
            k = k + 3

    mat_factorized = factorized(csc_matrix(mat))
    lhs = mat_factorized(rhs)

    lhsx = RHSx
    lhsx2 = [[lhs[i + 3 * j - np.count_nonzero(np.isnan(RHSx[:, :j]))] if ~np.isnan(lhsx[i, j]) else np.nan for j in
              range(n - 1)] for i in range(3)]

    U_re[1, :] = lhsx2[0][:]
    U_im[1, :] = lhsx2[1][:]
    Q[0, :] = lhsx2[2][:]

    U[1, :] = U_re[1, :] + U_im[1, :] * 1j
    X[1, :] = (-X[0, :] * np.conj(U[1, :])) / np.conj(U[0, :])
    X_re[1, :] = np.real(X[1, :])
    X_im[1, :] = np.imag(X[1, :])

    # .......................CALCULATION OF TERMS [1]. DONE

    # .......................CALCULATION OF TERMS [>=2]

    for c in range(2, n_coeff):  # c defines the current depth
        valor = np.zeros(n - 1, dtype=complex)
        valor[pq - 1] = (vec_P[pq - 1, 0] - vec_Q[pq - 1, 0] * 1j) * X[c - 1, pq - 1] + U[c - 1, pq - 1] * vec_shunts[
            pq - 1, 0]
        valor[pv - 1] = conv(X, Q, c, pv - 1, 2) * (-1) * 1j + U[c - 1, pv - 1] * vec_shunts[pv - 1, 0] + \
            X[c - 1, pv - 1] * vec_P[pv - 1, 0]

        RHSx = np.zeros((3, n - 1), dtype=float)  # a warning appears but that is not critical
        RHSx[0, pq - 1] = valor[pq - 1].real
        RHSx[1, pq - 1] = valor[pq - 1].imag
        RHSx[2, pq - 1] = np.nan  # dummy cell to later eliminate

        RHSx[0, pv - 1] = valor[pv - 1].real
        RHSx[1, pv - 1] = valor[pv - 1].imag
        RHSx[2, pv - 1] = -conv(U, U, c, pv - 1, 3)

        rhs = np.matrix.flatten(RHSx, 'f')
        rhs = rhs[~np.isnan(rhs)]

        # solve
        LHS = mat_factorized(rhs)

        LHSx = RHSx
        LHSx2 = [[LHS[i + 3 * j - np.count_nonzero(np.isnan(RHSx[:, :j]))] if ~np.isnan(LHSx[i, j]) else np.nan for j in
                  range(n - 1)] for i in range(3)]

        U_re[c, :] = LHSx2[0][:]
        U_im[c, :] = LHSx2[1][:]
        Q[c - 1, :] = LHSx2[2][:]

        U[c, :] = U_re[c, :] + U_im[c, :] * 1j
        X[c, range(n - 1)] = -conv(U, X, c, range(n - 1), 1) / np.conj(U[0, range(n - 1)])
        X_re[c, :] = np.real(X[c, :])
        X_im[c, :] = np.imag(X[c, :])

    # .......................CALCULATION OF TERMS [>=2]. DONE

    # sum the coefficients
    V = V0.copy()
    V[pqpv] = U.sum(axis=0)
    return V


if __name__ == '__main__':


    V2 = helm_josep(n,
                    Y,
                    vec_Y0,
                    V0,
                    S0,
                    vec_shunts,
                    pq,
                    pv,
                    vd,
                    n_coeff=30)