# AUTHORS: Santiago Peñate Vera and Josep Fanals Batllori
# CONTACT:  santiago.penate.vera@gmail.com, u1946589@campus.udg.edu
# thanks to Llorenç Fanals Batllori for his help at coding

# --------------------------- LIBRARIES
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import spsolve, factorized
np.set_printoptions(linewidth=2000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# --------------------------- END LIBRARIES

# --------------------------- INITIAL DATA: Y, SHUNTS AND Y0i
df_top = pd.read_excel('Data.xlsx', sheet_name='Topologia')  # DataFrame of the topology
df_bus = pd.read_excel('Data.xlsx', sheet_name='Busos')  # dataframe of the buses

n = df_bus.shape[0]  # number of buses, including slacks
nl = df_top.shape[0]  # number of lines

A = np.zeros((n, nl), dtype=int)  # incidence matrix
L = np.zeros((nl, nl), dtype=complex)
np.fill_diagonal(L, [1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) for i in range(nl)])
A[df_top.iloc[range(nl), 0], range(nl)] = 1  # buses names must be > 0 and integers
A[df_top.iloc[range(nl), 1], range(nl)] = -1
Ybus = np.dot(np.dot(A, L), np.transpose(A))
Ybus = csc_matrix(Ybus)


vec_Pi = np.zeros(n, dtype=float)  # data of active power
vec_Qi = np.zeros(n, dtype=float)  # data of reactive power
vec_Vi = np.zeros(n, dtype=float)  # data of voltage magnitude
vec_Wi = np.zeros(n, dtype=float)  # voltage magnitude squared


pq = []  # PQ buses indices
pv = []  # PV buses indices
sl = []  # Slack buses indices
vec_Pi[:] = df_bus.iloc[:, 1]
vec_Qi[:] = df_bus.iloc[:, 2]
vec_Vi[:] = df_bus.iloc[:, 3]
V_sl = []

for i in range(n):  # store the data of both PQ and PV
    if df_bus.iloc[i, 5] == 'PQ':
        pq.append(i)
    elif df_bus.iloc[i, 5] == 'PV':
        pv.append(i)
    elif df_bus.iloc[i, 5] == 'Slack':
        V_sl.append(df_bus.iloc[i, 3]*(np.cos(df_bus.iloc[i, 4])+np.sin(df_bus.iloc[i, 4])*1j))
        sl.append(i)

pqpv = np.sort(np.r_[pq, pv])
pqpv_x = pqpv
pq_x = pq
pv_x = pv
npq = len(pq)
npv = len(pv)
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

vec_shunts = vecx_shunts[pqpv]


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
prof = 10  # depth
U = np.zeros((prof, npqpv), dtype=complex)  # voltages
U_re = np.zeros((prof, npqpv), dtype=float)  # real part of voltages
U_im = np.zeros((prof, npqpv), dtype=float)  # imaginary part of voltages
X = np.zeros((prof, npqpv), dtype=complex)  # X=1/conj(U)
X_re = np.zeros((prof, npqpv), dtype=float)  # real part of X
X_im = np.zeros((prof, npqpv), dtype=float)  # imaginary part of X
Q = np.zeros((prof, npqpv), dtype=complex)  # unknown reactive powers
vec_W = vec_V * vec_V
dimensions = 2 * npq + 3 * npv  # number of unknowns
Yred = Ybus[np.ix_(pqpv, pqpv)]  # admittance matrix without slack buses
G = np.real(Yred)  # real parts of Yij
B = np.imag(Yred)  # imaginary parts of Yij

# indices 0 based
nsl_counted = np.zeros(n, dtype=int)
compt = 0
for i in range(n):
    if i in sl:
        compt += 1
    nsl_counted[i] = compt

pq = pq - nsl_counted[pq]
pv = pv - nsl_counted[pv]
pqpv = np.sort(np.r_[pq, pv])


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
pq = np.array(pq, dtype=int)
pv = np.array(pv, dtype=int)

prod = np.dot((Ysl[pqpv, :]), V_sl[:])

valor[pq] = prod[pq] - Ysl[pq].sum(axis=1) + (vec_P[pq] - vec_Q[pq] * 1j) * X[0, pq] + U[0, pq] * vec_shunts[pq, 0]
valor[pv] = prod[pv] - Ysl[pv].sum(axis=1) + (vec_P[pv]) * X[0, pv] + U[0, pv] * vec_shunts[pv, 0]


RHS = np.zeros(2*npqpv + npv, dtype=float)
RHS[pq] = valor[pq].real
RHS[pv] = valor[pv].real
RHS[npqpv + pq] = valor[pq].imag
RHS[npqpv + pv] = valor[pv].imag
RHS[2 * npqpv:] = vec_W[pv] - 1


MAT = lil_matrix((dimensions, dimensions), dtype=float)
MAT[:npqpv, :npqpv] = G
MAT[npqpv:2 * npqpv, :npqpv] = B
MAT[:npqpv, npqpv:2 * npqpv] = -B
MAT[npqpv:2 * npqpv, npqpv:2 * npqpv] = G

MAT_URE = np.zeros((npqpv, npqpv), dtype=float)
np.fill_diagonal(MAT_URE, 2 * U_re[0, :])
MAT[2 * npqpv:, :npqpv] = np.delete(MAT_URE, list(pq), axis=0)

MAT_UIM = np.zeros((npqpv, npqpv), dtype=float)
np.fill_diagonal(MAT_UIM, 2 * U_im[0, :])
MAT[2 * npqpv:, npqpv:2 * npqpv] = np.delete(MAT_UIM, list(pq), axis=0)

MAT_XIM = np.zeros((npqpv, npqpv), dtype=float)
np.fill_diagonal(MAT_XIM, -X_im[0, :])
MAT[:npqpv, 2 * npqpv:] = np.delete(MAT_XIM, list(pq), axis=1)

MAT_XRE = np.zeros((npqpv, npqpv), dtype=float)
np.fill_diagonal(MAT_XRE, X_re[0, :])
MAT[npqpv:2 * npqpv, 2 * npqpv:] = np.delete(MAT_XRE, list(pq), axis=1)

# factorize (only once)
MAT_csc = factorized(MAT.tocsc())

# solve
LHS = MAT_csc(RHS)

U_re[1, :] = LHS[:npqpv]
U_im[1, :] = LHS[npqpv:2 * npqpv]
Q[0, pv] = LHS[2 * npqpv:]

U[1, :] = U_re[1, :] + U_im[1, :] * 1j
X[1, :] = (-X[0, :] * np.conj(U[1, :])) / np.conj(U[0, :])
X_re[1, :] = X[1, :].real
X_im[1, :] = X[1, :].imag

# .......................CALCULATION OF TERMS [1]. DONE


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


for c in range(2, prof):  # c defines the current depth

    valor[pq] = (vec_P[pq] - vec_Q[pq] * 1j) * X[c - 1, pq] + U[c - 1, pq] * vec_shunts[pq, 0]
    valor[pv] = conv(X, Q, c, pv, 2) * -1j + U[c - 1, pv] * vec_shunts[pv, 0] + X[c - 1, pv] * vec_P[pv]

    RHS[pq] = valor[pq].real
    RHS[pv] = valor[pv].real
    RHS[npqpv + pq] = valor[pq].imag
    RHS[npqpv + pv] = valor[pv].imag
    RHS[2 * npqpv:] = -conv(U, U, c, pv, 3).real  # the convolution of 2 complex is real :)

    LHS = MAT_csc(RHS)

    U_re[c, :] = LHS[:npqpv]
    U_im[c, :] = LHS[npqpv:2 * npqpv]
    Q[c - 1, pv] = LHS[2 * npqpv:]

    U[c, :] = U_re[c, :] + U_im[c, :] * 1j
    X[c, range(npqpv)] = -conv(U, X, c, range(npqpv), 1) / np.conj(U[0, range(npqpv)])
    X_re[c, :] = np.real(X[c, :])
    X_im[c, :] = np.imag(X[c, :])
# .......................CALCULATION OF TERMS [>=2]. DONE

# --------------------------- CHECK DATA
U_final = np.zeros(npqpv, dtype=complex)  # final voltages
U_final[0:npqpv] = U.sum(axis=0)
I_serie = Yred * U_final  # current flowing through series elements
I_inj_slack = np.dot((Ysl[pqpv, :]), V_sl[:])
I_shunt = np.zeros(npqpv, dtype=complex)  # current through shunts
I_shunt[:] = -U_final * vec_shunts[:, 0]  # change the sign again
I_gen_out = I_serie - I_inj_slack + I_shunt  # current leaving the bus

# assembly the reactive power vector
Qfinal = vec_Q.copy()
Qfinal[pv] = (Q[:, pv] * 1j).sum(axis=0).imag

# compute the current injections
I_gen_in = (vec_P - Qfinal * 1j) / np.conj(U_final)

U_fi = np.zeros(n, dtype=complex)
Q_fi = np.zeros(n, dtype=complex)
P_fi = np.zeros(n, dtype=complex)
I_dif = np.zeros(n, dtype=complex)
S_dif = np.zeros(n, dtype=complex)

U_fi[pqpv_x] = U_final
U_fi[sl] = V_sl

Q_fi[pqpv_x] = Qfinal
Q_fi[sl] = np.nan

P_fi[pqpv_x] = vec_P
P_fi[sl] = np.nan

I_dif[pqpv_x] = I_gen_in - I_gen_out
I_dif[sl] = np.nan

S_dif[pqpv_x] = np.conj(I_gen_in - I_gen_out) * U_final
S_dif[sl] = np.nan

df = pd.DataFrame(np.c_[np.abs(U_fi), np.angle(U_fi) * 180 / np.pi, np.real(P_fi), np.real(Q_fi), np.abs(I_dif),
                        np.abs(S_dif)], columns=['|V|', 'Angle (deg)', 'P', 'Q', 'I error', 'S error'])

print(df)
# test













"""
V_test = np.array([0.95368602,
                   0.94166879,
                   0.93910714,
                   0.95,
                   0.94,
                   0.92973537,
                   0.93579263,
                   0.91,
                   0.94618528,
                   0.98,
                   0.92])

ok = np.isclose(abs(U_final), V_test, atol=1e-3).all()

if not ok:
    print('Test failed')
"""
