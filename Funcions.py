#PER NO COPIAR TOTES LES FUNCIONS A L'ARXIU PRINCIPAL. IMPORTAR-LES
import numpy as np
import numba as nb
from numba import jit
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse import lil_matrix, diags, hstack, vstack
from scipy.sparse.linalg import spsolve, factorized
from numpy import zeros, ones, mod, conj, array, r_, linalg, Inf, complex128, c_, r_, angle

@nb.jit
def pade4all(order, coeff_mat, s):
    """
    Computes the "order" Padè approximant of the coefficients at the approximation point s
    Arguments:
        coeff_mat: coefficient matrix (order, buses)
        order:  order of the series
        s: point of approximation
    Returns:
        Padè approximation at s for all the series
    """
    complex_type = nb.complex128
    nbus = coeff_mat.shape[1]
    voltages = np.zeros(nbus, complex_type)
    nn = int(order / 2)
    L = nn
    M = nn
    for d in range(nbus):
        rhs = coeff_mat[L + 1:L + M + 1, d]
        C = np.zeros((L, M), complex_type)
        for i in range(L):
            k = i + 1
            C[i, :] = coeff_mat[L - M + k:L + k, d]
        b = np.zeros(rhs.shape[0] + 1, complex_type)
        x = np.linalg.solve(C, -rhs)  # bn to b1
        b[0] = 1
        b[1:] = x[::-1]
        a = np.zeros(L + 1, complex_type)
        a[0] = coeff_mat[0, d]
        for i in range(L):
            val = complex(0)
            k = i + 1
            for j in range(k + 1):
                val += coeff_mat[k - j, d] * b[j]
            a[i + 1] = val
        p = complex(0)
        q = complex(0)
        for i in range(L + 1):
            p += a[i] * s ** i
            q += b[i] * s ** i
        voltages[d] = p / q
    return voltages

@nb.jit
def eta(U_inicial, limit):
    complex_type = nb.complex128
    n = limit
    Um = np.zeros(n, complex_type)
    Um[:] = U_inicial[:limit]
    mat = np.zeros((n, n+1), complex_type)
    mat[:, 0] = np.inf  # infinit
    mat[:, 1] = Um[:]
    for j in range(2, n + 1):
        if j % 2 == 0:
            for i in range(0, n + 1 - j):
                mat[i, j] = 1 / (1 / mat[i+1, j-2] + 1 / (mat[i+1, j-1]) - 1 / (mat[i, j-1]))
        else:
            for i in range(0, n + 1 - j):
                mat[i, j] = mat[i+1, j-2] + mat[i+1, j-1] - mat[i, j-1]
    return np.sum(mat[0, 1:])

@nb.jit
def aitken(U, limit):
    def S(Um, k):
        suma = 0
        for m in range(k+1):
            suma += Um[m]
        return suma
    complex_type = nb.complex128
    Um = U[:limit]  # només els 10 primers coeficients, si no, divideix per 0 i es deteriora
    n = limit
    T = np.zeros(n-2, complex_type)
    for i in range(len(T)):
        T[i] = S(Um, i) - (S(Um, i+1)**2 + S(Um, i)**2 - 2*S(Um, i+1)*S(Um, i)) / (S(Um, i+2) - 2*S(Um, i+1) + S(Um, i))
    return T[-1]  # l'últim element, entenent que és el que millor aproxima

@nb.jit
def theta(U_inicial, limit):
    def S(Um, k):
        suma = 0
        for m in range(k+1):
            suma = suma + Um[m]
        return suma
    complex_type = nb.complex128
    n = limit
    Um = np.zeros(n, complex_type)
    Um[:] = U_inicial[:limit]
    mat = np.zeros((n, n+1), complex_type)
    for i in range(n):
        mat[i, 1] = S(Um, i)  # plena de sumes parcials
    for j in range(2, n+1):
        if j % 2 == 0:
            for i in range(0, n+1-j):
                mat[i, j] = mat[i+1, j-2] + 1 / (mat[i+1, j-1] - mat[i, j-1])
        else:
            for i in range(0, n + 1 - j):
                mat[i, j] = mat[i+1, j-2] + ((mat[i+2, j-2] - mat[i+1, j-2]) * (mat[i+2, j-1] - mat[i+1, j-1])) \
                            / (mat[i+2, j-1] - 2 * mat[i+1, j-1] + mat[i, j-1])
    if limit % 2 == 0:
        return mat[0, n-1]
    else:
        return mat[0, n]

@nb.jit
def rho(U, limit):  # veure si cal tallar U, o sigui, agafar per exemple els 10 primers coeficients
    def S(Um, k):
        suma = 0
        for m in range(k+1):
            suma = suma + Um[m]
        return suma
    complex_type = nb.complex128
    Um = U[:limit]  # no agafar tots els coeficients, si no, salta error
    n = limit
    mat = np.zeros((n, n+1), complex_type)
    for i in range(n):
        mat[i, 1] = S(Um, i)  # plena de sumes parcials
    for j in range(2, n+1):
        for i in range(0, n+1-j):
            mat[i, j] = mat[i+1, j-2] + (j - 1) / (mat[i+1,j-1] - mat[i, j-1])
    if limit % 2 == 0:
        return mat[0, n-1]
    else:
        return mat[0, n]

@nb.jit
def epsilon2(U, limit):
    def S(Um, k):
        suma = 0
        for m in range(k+1):
            suma = suma + Um[m]
        return suma
    complex_type = nb.complex128
    Um = U[:limit]  # no agafar tots els coeficients, si no, salta error
    n = limit
    mat = np.zeros((n, n+1), complex_type)
    for i in range(n):
        mat[i, 1] = S(Um, i)  # plena de sumes parcials
    for j in range(2, n+1):
        for i in range(0, n+1-j):
            mat[i, j] = mat[i+1, j-2] + 1 / (mat[i+1, j-1] - mat[i, j-1])
    if limit % 2 == 0:
        return mat[0, n-1]
    else:
        return mat[0, n]

@nb.jit
def thevenin_funcX2(U, X, i):
    complex_type = nb.complex128
    n = len(U)
    r_3 = np. zeros(n, complex_type)
    r_2 = np. zeros(n, complex_type)
    r_1 = np. zeros(n, complex_type)
    r_0 = np. zeros(n, complex_type)
    T_03 = np. zeros(n, complex_type)
    T_02 = np. zeros(n, complex_type)
    T_01 = np. zeros(n, complex_type)
    T_00 = np. zeros(n, complex_type)
    T_13 = np. zeros(n, complex_type)
    T_12 = np. zeros(n, complex_type)
    T_11 = np. zeros(n, complex_type)
    T_10 = np. zeros(n, complex_type)
    T_23 = np. zeros(n, complex_type)
    T_22 = np. zeros(n, complex_type)
    T_21 = np. zeros(n, complex_type)
    T_20 = np. zeros(n, complex_type)

    #A LA NOVA MANERA, CONSIDERANT QUE U[0] POT SER DIFERENT D'1:
    r_0[0] = -1
    r_1[0:n - 1] = U[1:n] / U[0]
    r_2[0:n - 2] = U[2:n] / U[0] - U[1] * np.conj(U[0]) / U[0] * X[1:n - 1]

    T_00[0] = -1
    T_01[0] = -1
    T_02[0] = -1
    T_10[0] = 0
    T_11[0] = 1 / U[0]
    T_12[0] = 1 / U[0]
    T_20[0] = 0
    T_21[0] = 0
    T_22[0] = -U[1] * np.conj(U[0]) / U[0]


    for l in range(n):  # ANAR CALCULANT CONSTANTS , RESIDUS I POLINOMIS
        a = (r_2[0] * r_1[0]) / (- r_0[1] * r_1[0] + r_0[0] * r_1[1] - r_0[0] * r_2[0])
        b = -a * r_0[0] / r_1[0]
        c = 1 - b
        T_03[0] = b * T_01[0] + c * T_02[0]
        T_03[1:n] = a * T_00[0:n-1] + b * T_01[1:n] + c * T_02[1:n]
        T_13[0] = b * T_11[0] + c * T_12[0]
        T_13[1:n] = a * T_10[0:n-1] + b * T_11[1:n] + c * T_12[1:n]
        T_23[0] = b * T_21[0] + c * T_22[0]
        T_23[1:n] = a * T_20[0:n-1] + b * T_21[1:n] + c * T_22[1:n]
        r_3[0:n-2] = a * r_0[2:n] + b * r_1[2:n] + c * r_2[1:n-1]

        if l == n - 1:
            t_0 = T_03
            t_1 = T_13
            t_2 = T_23

        r_0[:] = r_1[:]
        r_1[:] = r_2[:]
        r_2[:] = r_3[:]
        T_00[:] = T_01[:]
        T_01[:] = T_02[:]
        T_02[:] = T_03[:]
        T_10[:] = T_11[:]
        T_11[:] = T_12[:]
        T_12[:] = T_13[:]
        T_20[:] = T_21[:]
        T_21[:] = T_22[:]
        T_22[:] = T_23[:]

        r_3 = np.zeros(n, complex_type)
        T_03 = np.zeros(n, complex_type)
        T_13 = np.zeros(n, complex_type)
        T_23 = np.zeros(n, complex_type)

    usw = -np.sum(t_0) / np.sum(t_1)
    sth = -np.sum(t_2) / np.sum(t_1)

    sigma_bo = sth / (usw * np.conj(usw))

    u = 0.5 + np.sqrt(0.25 + np.real(sigma_bo) - np. imag(sigma_bo)**2) + np.imag(sigma_bo)*1j  # positive branch
    #u = 0.5 - np.sqrt(0.25 + np.real(sigma_bo) - np.imag(sigma_bo) ** 2) + np.imag(sigma_bo) * 1j  # negative branch
    ufinal = u*usw

    return ufinal


def Sigma_funcO(coeff_matU, coeff_matX, order, V_slack):
    """
    :param coeff_matU: array with voltage coefficients
    :param coeff_matX: array with inverse conjugated voltage coefficients
    :param order: should be prof - 1
    :param V_slack: slack bus voltage vector. Must contain only 1 slack bus
    :return: sigma complex value
    """
    #complex_type = nb.complex128
    if len(V_slack) > 1:
        print('Sigma values may not be correct')
    V0 = V_slack[0]
    coeff_matU = coeff_matU / V0
    coeff_matX = coeff_matX / V0
    nbus = coeff_matU.shape[1]
    sigmes = np.zeros(nbus, dtype=complex)
    if order % 2 == 0:
        M = int(order / 2) - 1
    else:
        M = int(order / 2)
    for d in range(nbus):
        a = coeff_matU[1:2 * M + 2, d]
        b = coeff_matX[0:2 * M + 1, d]
        C = np.zeros((2 * M + 1, 2 * M + 1), dtype=complex)
        for i in range(2 * M + 1):
            if i < M:
                C[1 + i:, i] = a[:2 * M - i]
            else:
                C[i - M:, i] = - b[:3 * M - i + 1]
        lhs = np.linalg.solve(C, -a)
        sigmes[d] = np.sum(lhs[M:])/(np.sum(lhs[:M]) + 1)
    return sigmes