# AUTHOR: Josep Fanals Batllori
# CONTACT: u1946589@campus.udg.edu.

# --------------------------- LIBRARIES
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve, factorized
np.set_printoptions(linewidth=2000)
# --------------------------- END LIBRARIES

# --------------------------- INITIAL DATA: Y, SHUNTS AND Y0i
df_top = pd.read_excel('Data.xlsx', sheet_name='Topologia')  # DataFrame of the topology

busos_coneguts = []  # vector to store the indices of the found buses
[busos_coneguts.append(df_top.iloc[i, j]) for i in range(df_top.shape[0]) for j in range(0, 2) if
 df_top.iloc[i, j] not in busos_coneguts]
n = len(busos_coneguts)
n_linies = df_top.shape[0]

A = np.zeros((n, n_linies), dtype=int)  # núm busos, núm línies
L = np.zeros((n_linies, n_linies), dtype=complex)
np.fill_diagonal(L, [1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) for i in range(n_linies)])
A[df_top.iloc[range(n_linies), 0], range(n_linies)] = 1
A[df_top.iloc[range(n_linies), 1], range(n_linies)] = -1
Yx = np.dot(np.dot(A, L), np.transpose(A))

Y = np.zeros((n - 1, n - 1), dtype=complex)  # admittance matrix without slack bus
Y[:, :] = Yx[1:, 1:]

vecx_shunts = np.zeros((n, 1), dtype=complex)  # vector with shunt admittances
for i in range(df_top.shape[0]):  # passar per totes les files
    vecx_shunts[df_top.iloc[i, 0], 0] = vecx_shunts[df_top.iloc[i, 0], 0] + df_top.iloc[
        i, 4] * -1j  # B/2 is in column 4. The sign is changed here
    vecx_shunts[df_top.iloc[i, 1], 0] = vecx_shunts[df_top.iloc[i, 1], 0] + df_top.iloc[
        i, 4] * -1j  # B/2 is in column 4. The sign is changed here

vec_shunts = np.zeros((n - 1, 1), dtype=complex)  # same vector, just to adapt
for i in range(n - 1):
    vec_shunts[i, 0] = vecx_shunts[i + 1, 0]

# vec_shunts = --vec_shunts  # no need to change the sign, already done

vec_Y0 = np.zeros((n - 1, 1), dtype=complex)  # vector with admittances connecting to the slack

for i in range(df_top.shape[0]):  # go through all rows
    if df_top.iloc[i, 0] == 0:  # if slack in the first column
        vec_Y0[df_top.iloc[i, 1] - 1, 0] = 1 / (
                    df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)  # -1 so bus 1 goes to index 0
    elif df_top.iloc[i, 1] == 0:  # if slack in the second column
        vec_Y0[df_top.iloc[i, 0] - 1, 0] = 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)

G = np.real(Y)  # real parts of Yij
B = np.imag(Y)  # imaginary parts of Yij

# --------------------------- INITIAL DATA: Y, SHUNTS AND Y0i. DONE
# --------------------------- INITIAL DATA: BUSES INFORMATION
df_bus = pd.read_excel('Data.xlsx', sheet_name='Busos')  # dataframe of the buses
if df_bus.shape[0] != n:
    print('Error: número de busos de ''Topologia'' i de ''Busos'' no és igual')  # check if number of buses is coherent

num_busos_PQ = 0  # initialize number of PQ buses
num_busos_PV = 0  # initialize number of PV buses
vec_busos_PQ = np.zeros([0], dtype=int)  # vector to store the indices of PQ buses
vec_busos_PV = np.zeros([0], dtype=int)  # vector to store the indices of PV buses
vec_P = np.zeros((n - 1, 1), dtype=float)  # data of active power
vec_Q = np.zeros((n - 1, 1), dtype=float)  # data of reactive power
vec_V = np.zeros((n - 1, 1), dtype=float)  # data of voltage magnitude
vec_W = np.zeros((n - 1, 1), dtype=float)  # voltage magnitude squared

for i in range(df_bus.shape[0]):  # find the voltage specified for the slack
    if df_bus.iloc[i, 0] == 0:
        V_slack = df_bus.iloc[i, 3]
    else:
        V_slack = 1

for i in range(df_bus.shape[0]):  # store the data of both PQ and PV
    vec_P[df_bus.iloc[i, 0] - 1] = df_bus.iloc[i, 1]  # -1 to start at 0
    if df_bus.iloc[i, 4] == 'PQ':
        vec_Q[df_bus.iloc[i, 0] - 1] = df_bus.iloc[i, 2]  # -1 to start at 0
        vec_busos_PQ = np.append(vec_busos_PQ, df_bus.iloc[i, 0])
    elif df_bus.iloc[i, 4] == 'PV':
        vec_V[df_bus.iloc[i, 0] - 1] = df_bus.iloc[i, 3]  # -1 to start at 0
        vec_busos_PV = np.append(vec_busos_PV, df_bus.iloc[i, 0])
num_busos_PQ = len(vec_busos_PQ)
num_busos_PV = len(vec_busos_PV)

vec_W = vec_V**2
# --------------------------- INITIAL DATA: BUSES INFORMATION. DONE

# --------------------------- PREPARING IMPLEMENTATION
prof = 30  # depth
U = np.zeros((prof, n - 1), dtype=complex)  # voltages
U_re = np.zeros((prof, n - 1), dtype=complex)  # real part of voltages
U_im = np.zeros((prof, n - 1), dtype=complex)  # imaginary part of voltages
X = np.zeros((prof, n - 1), dtype=complex)  # X=1/conj(U)
X_re = np.zeros((prof, n - 1), dtype=complex)  # real part of X
X_im = np.zeros((prof, n - 1), dtype=complex)  # imaginary part of X
Q = np.zeros((prof, n - 1), dtype=complex)  # unknown reactive powers
pqpv = np.r_[vec_busos_PQ, vec_busos_PV]
pq = vec_busos_PQ
pv = vec_busos_PV
np.sort(pqpv)
npq = len(pq)
npv = len(pv)
dimensions = 2 * npq + 3 * npv  # number of unknowns

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
valor = np.zeros(n - 1, dtype=complex)
valor[pq - 1] = (V_slack - 1) * vec_Y0[pq - 1, 0] + (vec_P[pq - 1, 0] - vec_Q[pq - 1, 0] * 1j) * X[0, pq - 1] + U[
    0, pq - 1] * vec_shunts[pq - 1, 0]
valor[pv - 1] = (V_slack - 1) * vec_Y0[pv - 1, 0] + (vec_P[pv - 1, 0]) * X[0, pv - 1] + U[0, pv - 1] * vec_shunts[
    pv - 1, 0]

RHS = np.zeros(2*(n - 1) + npv, dtype=complex) #RHS ben muntat! ara canviar mat
RHS[pq - 1] = np.real(valor[pq - 1])
RHS[pv - 1] = np.real(valor[pv - 1])
RHS[n - 1 + (pq - 1)] = np.imag(valor[pq - 1])
RHS[n - 1 + (pv - 1)] = np.imag(valor[pv - 1])
RHS[2 * (n - 1):] = vec_W[pv - 1, 0] - 1
#print(RHS[2 * (n - 1):])


MAT = lil_matrix((dimensions, 2 * (n - 1) + npv), dtype=complex)
MAT[:(n - 1), :(n - 1)] = G
MAT[(n - 1):2 * (n - 1), :(n - 1)] = B
MAT[:(n - 1), (n - 1):2 * (n - 1)] = -B
MAT[(n - 1):2 * (n - 1), (n - 1):2 * (n - 1)] = G

MAT_URE = np.zeros((n - 1, n - 1), dtype=complex)
np.fill_diagonal(MAT_URE, 2*U_re[0, :])
MAT[2 * (n - 1):, :(n - 1)] = np.delete(MAT_URE, list(pq - 1), axis=0)

MAT_UIM = np.zeros((n - 1, n - 1), dtype=complex)
np.fill_diagonal(MAT_UIM, 2*U_im[0, :])
MAT[2 * (n - 1):, (n - 1):2 * (n - 1)] = np.delete(MAT_UIM, list(pq - 1), axis=0)

MAT_XIM = np.zeros((n - 1, n - 1), dtype=complex)
np.fill_diagonal(MAT_XIM, -X_im[0, :])
MAT[:(n - 1), 2 * (n - 1):] = np.delete(MAT_XIM, list(pq - 1), axis=1)

MAT_XRE = np.zeros((n - 1, n - 1), dtype=complex)
np.fill_diagonal(MAT_XRE, X_re[0, :])
MAT[(n-1):2 * (n - 1), 2 * (n - 1):] = np.delete(MAT_XRE, list(pq - 1), axis=1)

MAT_lil = lil_matrix(MAT)
MAT_csc = factorized(csc_matrix(MAT_lil))
LHS = MAT_csc(RHS)

U_re[1, :] = LHS[:(n - 1)]
U_im[1, :] = LHS[(n - 1):2 * (n - 1)]
Q[0, pv - 1] = LHS[2 * (n - 1):]

U[1, :] = U_re[1, :] + U_im[1, :] * 1j
X[1, :] = (-X[0, :] * np.conj(U[1, :])) / np.conj(U[0, :])
X_re[1, :] = np.real(X[1, :])
X_im[1, :] = np.imag(X[1, :])

# .......................CALCULATION OF TERMS [1]. DONE

# .......................CALCULATION OF TERMS [>=2]


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

    valor[pq - 1] = (vec_P[pq - 1, 0] - vec_Q[pq - 1, 0] * 1j) * X[c - 1, pq - 1] + U[c - 1, pq - 1] * vec_shunts[
        pq - 1, 0]
    valor[pv - 1] = conv(X, Q, c, pv - 1, 2) * (-1) * 1j + U[c - 1, pv - 1] * vec_shunts[pv - 1, 0] + \
        X[c - 1, pv - 1] * vec_P[pv - 1, 0]

    RHS[pq - 1] = np.real(valor[pq - 1])
    RHS[pv - 1] = np.real(valor[pv - 1])
    RHS[n - 1 + (pq - 1)] = np.imag(valor[pq - 1])
    RHS[n - 1 + (pv - 1)] = np.imag(valor[pv - 1])
    RHS[2 * (n - 1):] = -conv(U, U, c, pv - 1, 3)

    LHS = MAT_csc(RHS)

    U_re[c, :] = LHS[:(n - 1)]
    U_im[c, :] = LHS[(n - 1):2 * (n - 1)]
    Q[c - 1, pv - 1] = LHS[2 * (n - 1):]

    U[c, :] = U_re[c, :] + U_im[c, :] * 1j
    X[c, range(n - 1)] = -conv(U, X, c, range(n - 1), 1) / np.conj(U[0, range(n - 1)])
    X_re[c, :] = np.real(X[c, :])
    X_im[c, :] = np.imag(X[c, :])
# .......................CALCULATION OF TERMS [>=2]. DONE

# --------------------------- CHECK DATA
U_final = np.zeros(n - 1, dtype=complex)  # final voltages
U_final[0:n - 1] = U.sum(axis=0)
I_serie = Y * U_final  # current flowing through series elements
I_inj_slack = vec_Y0[:, 0] * V_slack
I_shunt = np.zeros((n - 1), dtype=complex)  # current through shunts
I_shunt[:] = -U_final[:] * vec_shunts[:, 0]  # change the sign again
I_generada = I_serie - I_inj_slack + I_shunt  # current leaving the bus
I_gen2 = [(vec_P[i, 0] - vec_Q[i, 0] * 1j) / np.conj(U_final[i]) if i + 1 in pq else
          (vec_P[i, 0] - sum(Q[:, i] * 1j)) / np.conj(U_final[i]) for i in range(n - 1)]

print(U_final)
print(abs(U_final))
print(I_gen2 - I_generada)  # current balance. Should be almost 0
Qdf = pd.DataFrame(Q)  # to check the unknown reactive power
Qdf.to_excel('Results_reactive_power.xlsx', index=False, header=False)
Udf = pd.DataFrame(U)
Udf.to_excel('Results_voltage_coefficients.xlsx', index=False, header=False)  # to check the voltages