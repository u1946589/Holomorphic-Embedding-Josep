
import numpy as np
import numpy as np
from mpmath import *
mp.dps = 30; mp. pretty = False

def thevenin_funcX2(U, X, i):
    n = len(U)
    r_3 = np. zeros(n, dtype=complex)
    r_2 = np. zeros(n, dtype=complex)
    r_1 = np. zeros(n, dtype=complex)
    r_0 = np. zeros(n, dtype=complex)
    T_03 = np. zeros(n, dtype=complex)
    T_02 = np. zeros(n, dtype=complex)
    T_01 = np. zeros(n, dtype=complex)
    T_00 = np. zeros(n, dtype=complex)
    T_13 = np. zeros(n, dtype=complex)
    T_12 = np. zeros(n, dtype=complex)
    T_11 = np. zeros(n, dtype=complex)
    T_10 = np. zeros(n, dtype=complex)
    T_23 = np. zeros(n, dtype=complex)
    T_22 = np. zeros(n, dtype=complex)
    T_21 = np. zeros(n, dtype=complex)
    T_20 = np. zeros(n, dtype=complex)

    r_0[0] = -1
    r_1[0:n-1] = U[1:n]
    r_2[0:n-2] = U[2:n] - U[1] * X[1:n-1]

    T_00[0] = -1
    T_01[0] = -1
    T_02[0] = -1
    T_10[0] = 0
    T_11[0] = 1
    T_12[0] = 1
    T_20[0] = 0
    T_21[0] = 0
    T_22[0] = -U[1]

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

        r_3 = np.zeros(n, dtype=complex)
        T_03 = np.zeros(n, dtype=complex)
        T_13 = np.zeros(n, dtype=complex)
        T_23 = np.zeros(n, dtype=complex)

    usw = -sum(t_0) / sum(t_1)
    sth = -sum(t_2) / sum(t_1)

    sigma_bo = sth / (usw * np.conj(usw))

    u = 0.5 + np.sqrt(0.25 + np.real(sigma_bo) - np. imag(sigma_bo)**2) + np.imag(sigma_bo)*1j  # positive branch
    #u = 0.5 - np.sqrt(0.25 + np.real(sigma_bo) - np.imag(sigma_bo) ** 2) + np.imag(sigma_bo) * 1j  # negative branch
    ufinal = u*usw

    return ufinal

