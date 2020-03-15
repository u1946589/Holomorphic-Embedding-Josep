# AUTHORS: Santiago Peñate Vera and Josep Fanals Batllori
# CONTACT:  santiago.penate.vera@gmail.com, u1946589@campus.udg.edu
# thanks to Llorenç Fanals Batllori for his help at coding
# per a testejar els errors dels shunts

# --------------------------- LIBRARIES
import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse import lil_matrix, diags, hstack, vstack
from scipy.sparse.linalg import spsolve, factorized
from numpy import zeros, ones, mod, conj, array, r_, linalg, Inf, complex128, c_, r_, angle
np.set_printoptions(linewidth=2000, edgeitems=1000, suppress=True)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 2000)
pd.set_option("display.precision", 6)
# --------------------------- END LIBRARIES


@nb.njit("(c16[:])(i8, c16[:, :], i8)")
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
    nbus = coeff_mat.shape[1]
    complex_type = nb.complex128
    voltages = np.zeros(nbus, dtype=complex_type)
    nn = int(order / 2)
    L = nn
    M = nn
    for d in range(nbus):
        rhs = coeff_mat[L + 1:L + M + 1, d]
        C = np.zeros((L, M), dtype=complex_type)
        for i in range(L):
            k = i + 1
            C[i, :] = coeff_mat[L - M + k:L + k, d]
        b = np.zeros(rhs.shape[0] + 1, dtype=complex_type)
        x = np.linalg.solve(C, -rhs)  # bn to b1
        b[0] = 1
        b[1:] = x[::-1]
        a = np.zeros(L + 1, dtype=complex_type)
        a[0] = coeff_mat[0, d]
        for i in range(L):
            val = complex_type(0)
            k = i + 1
            for j in range(k + 1):
                val += coeff_mat[k - j, d] * b[j]
            a[i + 1] = val
        p = complex_type(0)
        q = complex_type(0)
        for i in range(L + 1):
            p += a[i] * s ** i
            q += b[i] * s ** i
        voltages[d] = p / q
    return voltages

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

def rho(U):  # veure si cal tallar U, o sigui, agafar per exemple els 10 primers coeficients
    def S(U, k):
        suma = 0
        for m in range(k+1):
            suma = suma + U[m]
        return suma
    U = U[:20]  # no agafar tots els coeficients, si no, salta error
    n = len(U)
    mat = np.zeros((n, n+1), dtype=complex)
    #mat[:, 0] = 0  # plena de 0, tot i que ja ho està
    for i in range(n):
        mat[i, 1] = S(U, i)  # plena de sumes parcials
    for j in range(2, n+1):
        for i in range(0, n+1-j):
            mat[i, j] = mat[i+1, j-2] + (j - 1) / (mat[i+1,j-1] - mat[i, j-1])
    #print(mat)
    return mat[0, n-1]  # provar si és aquest número, o si depèn de parell o imparell...

def theta(U_inicial):
    def S(U, k):
        suma = 0
        for m in range(k+1):
            suma = suma + U[m]
        return suma
    n = len(U_inicial)
    U = np.zeros(n, dtype=complex)
    U[:] = U_inicial[:]
    mat = np.zeros((n, n+1), dtype=complex)
    for i in range(n):
        mat[i, 1] = S(U, i)  # plena de sumes parcials
    for j in range(2, n+1):
        if j % 2 == 0:
            for i in range(0, n+1-j):
                mat[i, j] = mat[i+1, j-2] + 1 / (mat[i+1, j-1] - mat[i, j-1])
        else:
            for i in range(0, n + 1 - j):
                mat[i, j] = mat[i+1, j-2] + ((mat[i+2, j-2] - mat[i+1, j-2]) * (mat[i+2, j-1] - mat[i+1, j-1])) \
                            / (mat[i+2, j-1] - 2 * mat[i+1, j-1] + mat[i, j-1])
    return mat[0, n-1]  # mirar si n està bé

def aitken(U):
    def S(U, k):
        suma = 0
        for m in range(k+1):
            suma = suma + U[m]
        return suma
    U = U[:10]  # només els 10 primers coeficients, si no, divideix per 0 i es deteriora
    n = len(U)
    T = np.zeros(n-2, dtype = complex)
    for i in range(len(T)):
        T[i] = S(U, i) - (S(U, i+1)**2 + S(U, i)**2 - 2*S(U, i+1)*S(U, i)) / (S(U, i+2) - 2*S(U, i+1) + S(U, i))
    return T[-1]  # l'últim element, entenent que és el que millor aproxima

def eta(U_inicial):
    n = len(U_inicial)
    U = np.zeros(n, dtype=complex)
    U[:] = U_inicial[:]
    mat = np.zeros((n, n+1), dtype=complex)
    mat[:, 0] = np.inf  # infinit
    mat[:, 1] = U[:]

    for j in range(2, n + 1):
        if j % 2 == 0:
            for i in range(0, n + 1 - j):
                mat[i, j] = 1 / (1 / mat[i+1, j-2] + 1 / (mat[i+1, j-1]) - 1 / (mat[i, j-1]))
        else:
            for i in range(0, n + 1 - j):
                mat[i, j] = mat[i+1, j-2] + mat[i+1, j-1] - mat[i, j-1]
    return sum(mat[0, 1:])

def epsilon(Sn, n, E):
    """
    Fast recursive Wynn's epsilon algorithm from:
        NONLINEAR SEQUENCE TRANSFORMATIONS FOR THE ACCELERATION OF CONVERGENCE
        AND THE SUMMATION OF DIVERGENT SERIES
        by Ernst Joachim Weniger
    Args:
        Sn: sum of coefficients
        n: order
        E: Coefficients structure copy that is modified in this algorithm
    Returns:
    """
    complex_type = complex128
    Zero = complex_type(0)
    One = complex_type(1)
    Tiny = np.finfo(complex_type).min
    Huge = np.finfo(complex_type).max

    E[n] = Sn

    if n == 0:
        estim = Sn
    else:
        AUX2 = Zero

        for j in range(n, 0, -1):  # range from n to 1 (both included)
            AUX1 = AUX2
            AUX2 = E[j-1]
            DIFF = E[j] - AUX2

            if abs(DIFF) <= Tiny:
                E[j-1] = Huge
            else:
                if DIFF == 0:
                    DIFF = Tiny
                E[j-1] = AUX1 + One / DIFF

        estim = E[n]

    return estim

#@nb.njit("(c8)(c16[:, :], c16[:, :], i8, i8, i8)")
def conv(A, B, c, i, tipus):
    if tipus == 1:
        suma = [np.conj(A[k, i]) * B[c - k, i] for k in range(1, c + 1)]
        return sum(suma)
    elif tipus == 2:
        suma = [A[k, i] * B[c - 1 - k, i] for k in range(1, c)]
        return sum(suma)
    elif tipus == 3:
        suma = [A[k, i] * np.conj(B[c - k, i]) for k in range(1, c)]
        return sum(suma)


# --------------------------- INITIAL DATA: Y, SHUNTS AND Y0i
df_top = pd.read_excel('Nord.xlsx', sheet_name='Topologia')  # DataFrame of the topology
df_bus = pd.read_excel('Nord.xlsx', sheet_name='Busos')  # Dataframe of the buses

n = df_bus.shape[0]  # number of buses, including slacks
nl = df_top.shape[0]  # number of lines

A = np.zeros((n, nl), dtype=int)  # incidence matrix
L = np.zeros((nl, nl), dtype=complex)
np.fill_diagonal(L, [1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) for i in range(nl)])
A[df_top.iloc[range(nl), 0], range(nl)] = 1  # buses names must be >= 0 and integers
A[df_top.iloc[range(nl), 1], range(nl)] = -1

Yseries = np.dot(np.dot(A, L), np.transpose(A))

Yseries_real = np.zeros((n,n), dtype=complex)
Yseries_real[:,:] = Yseries_real[:,:]

for i in range(nl):  # passejo per totes les línies
    tap = df_top.iloc[i, 5]
    if tap != 1:
        # element Ys/c**2
        Yseries[df_top.iloc[i, 0], df_top.iloc[i, 0]] += -1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) + 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) / (tap ** 2)
        # in fact, remains the same
        Yseries[df_top.iloc[i, 1], df_top.iloc[i, 1]] += -1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) + 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)
        # out of the diagonal element, -Ys/conj(c)
        Yseries[df_top.iloc[i, 0], df_top.iloc[i, 1]] += +1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) + -1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) / np.conj(tap)
        # out of the diagonal element, -Ys/c
        Yseries[df_top.iloc[i, 1], df_top.iloc[i, 0]] += +1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) + -1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) / tap

Yseries = csc_matrix(Yseries)
Yseries_real = csc_matrix(Yseries_real)


print('Yseries')
print(Yseries.toarray())

vec_Pi = np.zeros(n, dtype=float)  # data of active power
vec_Qi = np.zeros(n, dtype=float)  # data of reactive power
vec_Vi = np.zeros(n, dtype=float)  # data of voltage magnitude
vec_Wi = np.zeros(n, dtype=float)  # voltage magnitude squared


pq = []  # PQ buses indices
pv = []  # PV buses indices
sl = []  # Slack buses indices
vec_Pi[:] = np.nan_to_num(df_bus.iloc[:, 1])
vec_Qi[:] = np.nan_to_num(df_bus.iloc[:, 2])
vec_Vi[:] = np.nan_to_num(df_bus.iloc[:, 3])
V_sl = []

for i in range(n):  # store the data of both PQ and PV
    if df_bus.iloc[i, 5] == 'PQ':
        pq.append(i)
    elif df_bus.iloc[i, 5] == 'PV':
        pv.append(i)
    elif df_bus.iloc[i, 5] == 'Slack':
        V_sl.append(df_bus.iloc[i, 3]*(np.cos(df_bus.iloc[i, 4])+np.sin(df_bus.iloc[i, 4])*1j))
        sl.append(i)
pq = np.array(pq)
pv = np.array(pv)
npq = len(pq)
npv = len(pv)
if npv > 0 and npq > 0:
    pqpv = np.sort(np.r_[pq, pv])
elif npq > 0:
    pqpv = np.sort(pq)
elif npv > 0:
    pqpv = np.sort(pv)
pq_x = pq
pv_x = pv

npqpv = npq + npv
nsl = n - npqpv

vec_P = vec_Pi[pqpv]  # these 3 vectors are needed during the implementation
vec_Q = vec_Qi[pqpv]
vec_V = vec_Vi[pqpv]


vecx_shunts = np.zeros((n, 1), dtype=complex)  # vector with shunt admittances
for i in range(nl):  # go through all rows
    vecx_shunts[df_top.iloc[i, 0], 0] = vecx_shunts[df_top.iloc[i, 0], 0] + df_top.iloc[
        i, 4] * -1j  # B/2 is in column 4. The sign is changed here
    vecx_shunts[df_top.iloc[i, 1], 0] = vecx_shunts[df_top.iloc[i, 1], 0] + df_top.iloc[
        i, 4] * -1j  # B/2 is in column 4. The sign is changed here

for i in range(n):
    if df_bus.iloc[i, 6] != 0:
        vecx_shunts[df_bus.iloc[i, 0], 0] += df_bus.iloc[i, 6] * -1j

vec_shunts = vecx_shunts[pqpv]


df = pd.DataFrame(data=np.c_[vecx_shunts.imag, vec_Pi, vec_Qi, vec_Vi],
                  columns=['Ysh', 'P0', 'Q0', 'V0'])

print(df)

Yslx = np.zeros((n, n), dtype=complex)  # vector with admittances connecting to the slack buses

for i in range(nl):  # go through all rows
    if df_top.iloc[i, 0] in sl:  # if slack in the first column
        Yslx[df_top.iloc[i, 1], df_top.iloc[i, 0]] = 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) + \
                                                     Yslx[df_top.iloc[i, 1], df_top.iloc[i, 0]]
    elif df_top.iloc[i, 1] in sl:  # if slack in the second column
        Yslx[df_top.iloc[i, 0], df_top.iloc[i, 1]] = 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) + \
                                                     Yslx[df_top.iloc[i, 0], df_top.iloc[i, 1]]


Ysl1 = Yslx[:, sl]
Ysl = Ysl1[pqpv, :]


# --------------------------- INITIAL DATA: BUSES INFORMATION. DONE

# --------------------------- PREPARING IMPLEMENTATION
prof = 300  # depth
U = np.zeros((prof, npqpv), dtype=complex)  # voltages
U_re = np.zeros((prof, npqpv), dtype=float)  # real part of voltages
U_im = np.zeros((prof, npqpv), dtype=float)  # imaginary part of voltages
X = np.zeros((prof, npqpv), dtype=complex)  # X=1/conj(U)
X_re = np.zeros((prof, npqpv), dtype=float)  # real part of X
X_im = np.zeros((prof, npqpv), dtype=float)  # imaginary part of X
Q = np.zeros((prof, npqpv), dtype=complex)  # unknown reactive powers
vec_W = vec_V * vec_V
dimensions = 2 * npq + 3 * npv  # number of unknowns
Yred = Yseries[np.ix_(pqpv, pqpv)]  # admittance matrix without slack buses
Yred_real = Yseries_real[np.ix_(pqpv, pqpv)]
#Ydif = Yred_real-Yred
G = np.real(Yred)  # real parts of Yij
B = np.imag(Yred)  # imaginary parts of Yij

# indices 0 based in the reduced scheme
nsl_counted = np.zeros(n, dtype=int)
compt = 0
for i in range(n):
    if i in sl:
        compt += 1
    nsl_counted[i] = compt

if npv > 0 and npq > 0:
    pq_ = pq - nsl_counted[pq]
    pv_ = pv - nsl_counted[pv]
    pqpv_ = np.sort(np.r_[pq_, pv_])
elif npq > 0:
    pq_ = pq - nsl_counted[pq]
    pqpv_ = np.sort(pq_)
elif npv > 0:
    pv_ = pv - nsl_counted[pv]
    pqpv_ = np.sort(pv_)

# .......................CALCULATION OF TERMS [0]


if nsl > 1:
    U[0, :] = spsolve(Yred, Ysl.sum(axis=1))
else:
    U[0, :] = spsolve(Yred, Ysl)


X[0, :] = 1 / np.conj(U[0, :])
U_re[0, :] = U[0, :].real
U_im[0, :] = U[0, :].imag
X_re[0, :] = X[0, :].real
X_im[0, :] = X[0, :].imag

# .......................CALCULATION OF TERMS [0]. DONE

# .......................CALCULATION OF TERMS [1]
valor = np.zeros(npqpv, dtype=complex)

prod = np.dot((Ysl[pqpv_, :]), V_sl[:])



if npq > 0:
    valor[pq_] = prod[pq_] - Ysl[pq_].sum(axis=1) + (vec_P[pq_] - vec_Q[pq_] * 1j) * X[0, pq_] + U[0, pq_] * vec_shunts[pq_, 0]
if npv > 0:
    valor[pv_] = prod[pv_] - Ysl[pv_].sum(axis=1) + (vec_P[pv_]) * X[0, pv_] + U[0, pv_] * vec_shunts[pv_, 0]
    # compose the right-hand side vector
    RHS = np.r_[valor.real,
                valor.imag,
                vec_W[pv_] - np.real(U[0, pv_] * np.conj(U[0, pv_]))]
    #np.real(U[0, pv_] * np.conj(U[0, pv_]))
    # Form the system matrix (MAT)
    VRE = coo_matrix((2 * U_re[0, pv_], (np.arange(npv), pv_)), shape=(npv, npqpv)).tocsc()
    VIM = coo_matrix((2 * U_im[0, pv_], (np.arange(npv), pv_)), shape=(npv, npqpv)).tocsc()
    XIM = coo_matrix((-X_im[0, pv_], (pv_, np.arange(npv))), shape=(npqpv, npv)).tocsc()
    XRE = coo_matrix((X_re[0, pv_], (pv_, np.arange(npv))), shape=(npqpv, npv)).tocsc()
    EMPTY = csc_matrix((npv, npv))

    MAT = vstack((hstack((G,   -B,   XIM)),
                  hstack((B,    G,   XRE)),
                  hstack((VRE,  VIM, EMPTY))), format='csc')

else:
    # compose the right-hand side vector
    RHS = np.r_[valor.real,
                valor.imag]
    MAT = vstack((hstack((G, -B)),
                  hstack((B, G))), format='csc')

print('MAT')
print(MAT.toarray())

# factorize (only once)
MAT_LU = factorized(MAT.tocsc())

# solve
LHS = MAT_LU(RHS)

U_re[1, :] = LHS[:npqpv]
U_im[1, :] = LHS[npqpv:2 * npqpv]
if npv > 0:
    Q[0, pv_] = LHS[2 * npqpv:]

U[1, :] = U_re[1, :] + U_im[1, :] * 1j
X[1, :] = (-X[0, :] * np.conj(U[1, :])) / np.conj(U[0, :])
X_re[1, :] = X[1, :].real
X_im[1, :] = X[1, :].imag

# .......................CALCULATION OF TERMS [1]. DONE

range_pqpv = np.arange(npqpv)  # range of pqpv buses for the X coefficient

for c in range(2, prof):  # c defines the current depth
    if npq > 0:
        valor[pq_] = (vec_P[pq_] - vec_Q[pq_] * 1j) * X[c - 1, pq_] + U[c - 1, pq_] * vec_shunts[pq_, 0]
    if npv > 0:
        valor[pv_] = conv(X, Q, c, pv_, 2) * -1j + U[c - 1, pv_] * vec_shunts[pv_, 0] + X[c - 1, pv_] * vec_P[pv_]
        RHS = np.r_[valor.real,
                    valor.imag,
                    -conv(U, U, c, pv_, 3).real]
    else:
        RHS = np.r_[valor.real,
                    valor.imag]

    LHS = MAT_LU(RHS)

    U_re[c, :] = LHS[:npqpv]
    U_im[c, :] = LHS[npqpv:2 * npqpv]
    if npv > 0:
        Q[c - 1, pv_] = LHS[2 * npqpv:]

    U[c, :] = U_re[c, :] + 1j * U_im[c, :]
    X[c, range_pqpv] = -conv(U, X, c, range_pqpv, 1) / np.conj(U[0, range_pqpv])
    X_re[c, :] = np.real(X[c, :])
    X_im[c, :] = np.imag(X[c, :])
# .......................CALCULATION OF TERMS [>=2]. DONE

print('V coefficients')
print(U)
Ux = np.zeros((prof, npqpv), dtype=complex)  # voltages
Ux[:,:] = U[:,:]
Uxfi = np.zeros((prof, npqpv), dtype=complex)  # voltages
Uxfi[:,:] = U[:,:]
# --------------------------- CHECK DATA
U_final = np.zeros(npqpv, dtype=complex)  # final voltages
U_final[0:npqpv] = U.sum(axis=0)
I_serie = Yred * U_final  # current flowing through series elements
I_inj_slack = np.dot((Ysl[pqpv_, :]), V_sl[:])
I_shunt = np.zeros(npqpv, dtype=complex)  # current through shunts
I_shunt[:] = -U_final * vec_shunts[:, 0]  # change the sign again
I_gen_out = I_serie - I_inj_slack + I_shunt  # current leaving the bus

# assembly the reactive power vector
Qfinal = vec_Q.copy()
if npv > 0:
    Qfinal[pv_] = (Q[:, pv_] * 1j).sum(axis=0).imag

# compute the current injections
I_gen_in = (vec_P - Qfinal * 1j) / np.conj(U_final)

U_fi = np.zeros(n, dtype=complex)
Q_fi = np.zeros(n, dtype=complex)
P_fi = np.zeros(n, dtype=complex)
I_dif = np.zeros(n, dtype=complex)
S_dif = np.zeros(n, dtype=complex)
Sig_re = np.zeros(n, dtype=complex)
Sig_im = np.zeros(n, dtype=complex)
U_pa = np.zeros(n, dtype=complex)
U_sig = np.zeros(n, dtype=complex)
U_th = np.zeros(n, dtype=complex)
U_ait = np.zeros(n, dtype=complex)
U_eps = np.zeros(n, dtype=complex)
U_rho = np.zeros(n, dtype=complex)
U_theta = np.zeros(n, dtype=complex)
U_eta = np.zeros(n, dtype=complex)

U_fi[pqpv] = U_final
U_fi[sl] = V_sl

Q_fi[pqpv] = Qfinal
Q_fi[sl] = np.nan

P_fi[pqpv] = vec_P
P_fi[sl] = np.nan

I_dif[pqpv] = I_gen_in - I_gen_out
I_dif[sl] = np.nan

S_dif[pqpv] = np.conj(I_gen_in - I_gen_out) * U_final
S_dif[sl] = np.nan

U_pade = pade4all(prof - 1, U, 1)
V_slx = np.array(V_sl)

#Sigma = [Sigma_func(U[:,i], prof - 1, V_slx) for i in range(npqpv)]

U_pa[pqpv] = U_pade
U_pa[sl] = np.nan


U_sigx = U_pade / V_sl[0]
Sigmax = (U_sigx - 1) * np.conj(U_sigx)  # càlcul directe, avaluada a s=1


Sig_re[pqpv] = np.real(Sigmax)
Sig_im[pqpv] = np.imag(Sigmax)
Sig_re[sl] = np.nan
Sig_im[sl] = np.nan

"""
for i in range(npqpv):
    U_th[i] = thevenin_funcX2(U[:20, i], X[:20, i], 1)  # agafar pocs termes, si no, salta error
"""
for i in pq_:
    U_th[i] = thevenin_funcX2(U[:, i], X[:, i], 1)  # agafar pocs termes si el bus està poc arregat, si no, error
U_th[pv_] = np.nan
U_th[pqpv] = U_th[pqpv_]
U_th[sl] = np.nan


#AITKEN. FUNCIONA
for i in range(npqpv):
    U_ait[i] = aitken(U[:, i])

U_ait[pqpv] = U_ait[pqpv_]
U_ait[sl] = np.nan
#FI AITKEN

Ux2 = np.zeros((prof, npqpv), dtype=complex)  # cal tornar a agafar els valors perquè a alguna funció tallo la sèrie
Ux2[:,:] = Ux[:,:]

#EPSILON ACCELERADES
for i in range(npqpv):
    U_eps[i] = epsilon(sum(Ux[:, i]), 10, Ux[:, i])  # no agafar tots els coeficients, salta error
U_eps[pqpv] = U_eps[pqpv_]  # treballo amb Ux perquè U sembla canviar a algun moment
U_eps[sl] = np.nan
#FI EPSILON ACCELERADES

#RHO
for i in range(npqpv):
    U_rho[i] = rho(U[:, i])
U_rho[pqpv] = U_rho[pqpv_]
U_rho[sl] = np.nan
#print(U_rho)
#FI RHO

#THETA
for i in range(npqpv):
    U_theta[i] = theta(Ux2[:10, i])  # per no passar-li tots els coeficients, saltaria error per dividir / 0
U_theta[pqpv] = U_theta[pqpv_]
U_theta[sl] = np.nan
#print(U_theta)
#FI THETA

#ETA
for i in range(npqpv):
    U_eta[i] = eta(Ux2[:20, i])  # per no passar-li tots els coeficients, saltaria error per dividir / 0
U_eta[pqpv] = U_eta[pqpv_]
U_eta[sl] = np.nan
#print(abs(U_eta))
#FI ETA

s_pos = 1 / (2 * (abs(np.real(Sig_re) + np.real(Sig_im) * 1j) - np.real(Sig_re)))  #  és la s+
s_neg = - 1 / (2 * (abs(np.real(Sig_re) + np.real(Sig_im) * 1j) + np.real(Sig_re)))  #  és la s-


#IGUAL PERÒ SENSE ERRORS D'INTENSITAT
df = pd.DataFrame(np.c_[np.abs(U_fi), np.angle(U_fi), np.abs(U_pa), np.angle(U_pa),
                        np.abs(U_th), np.abs(U_eps), np.angle(U_eps), np.abs(U_ait), np.angle(U_ait), np.abs(U_rho),
                        np.angle(U_rho), np.abs(U_theta), np.angle(U_theta), np.abs(U_eta), np.angle(U_eta),
                        np.real(P_fi), np.real(Q_fi), np.abs(S_dif), np.real(Sig_re), np.real(Sig_im),
                        0.25+np.real(Sig_re)-np.real(Sig_im) ** 2, s_pos, s_neg],
                        columns=['|V| sum', 'A. sum', '|V| Padé', 'A. Padé',
                                 '|V| Thévenin', '|V| Epsilon', 'A. Epsilon', '|V| Aitken', 'A. Aitken',
                                 '|V| Rho', 'A. Rho', '|V| Theta', 'A. Theta', '|V| Eta', 'A. Eta', 'P', 'Q',
                                 'S error', 'Sigma re', 'Sigma im', 'Arrel', 's+', 's-'])

print(df)

Ybus = Yseries - diags(vecx_shunts[:, 0])

U_pa[sl] = V_sl

Scalc = U_pa * np.conj(Ybus * U_pa)  # assumint que Padé és el que dóna millor
S0 = np.real(P_fi) + 1j * np.real(Q_fi)
diff = S0 - Scalc
if npq > 0:
    err = max(abs(np.r_[diff[pqpv].real, diff[pq].imag]))
else:
    err = max(abs(np.r_[diff[pqpv].real]))
print('Power mismatch:', err)


#SIGMA, s a la que convergeix, Domb-Sykes

a=[]
b=[]
c=[]

x = np.linspace(-0.25, 1, 1000)
y = np.sqrt(0.25+x)
a.append(x)
b.append(y)
c.append(-y)



plt.plot(np.real(Sig_re), np.real(Sig_im), 'ro', markersize=2)
plt.plot(x, y)
plt.plot(x, -y)
plt.ylabel('Sigma im')
plt.xlabel('Sigma re')
plt.title('Gràfic Sigma')
plt.show()

#print(U[:20, :])

from Pade_Weierstrass_v1 import s_conv, valor_num, valor_vec
sx = np.zeros(npqpv)
#print(npqpv)

for i in range(npqpv):
    sx[i] = s_conv(U[:, i], 0.00000000001)

print(sx)

bb = np.zeros((prof, npqpv), dtype=complex)  # voltages
print(Uxfi[25,1])

for j in range(npqpv):
    for i in range(3, len(U)-1):
        bb[i, j] = np.abs(np.sqrt((Uxfi[i+1, j] * Uxfi[i-1, j] - Uxfi[i, j] ** 2) / (Uxfi[i, j] * Uxfi[i-2, j] -
                                                                              Uxfi[i-1, j] ** 2)))
        #bb[i, j] = (Uxfi[i, j]) / (Uxfi[i-1, j])


vec_1n = np.zeros(prof)
for i in range(3, prof):
    vec_1n[i] = 1 / i
    #vec_1n[i] = i
print(vec_1n)

plt.plot(vec_1n[50:len(U)], abs(bb[50:len(U), 29]), 'ro', markersize=2)

plt.show()

print(abs(bb[prof-2, 29]))
print(abs(bb[50:, 29]))
print(vec_1n[50:])

writer = pd.ExcelWriter('DombSykes.xlsx', engine='xlsxwriter')

df1 = pd.DataFrame(np.c_[abs(bb[50:, 29]), vec_1n[50:]])



df1.to_excel(writer, header=False, index=False)


writer.save()