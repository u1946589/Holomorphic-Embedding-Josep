# AUTHOR: Josep Fanals Batllori
# CONTACT: u1946589@campus.udg.edu.

# --------------------------- LIBRARIES
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix, diags
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
Ybus = np.dot(np.dot(A, L), np.transpose(A))
Ybus = csc_matrix(Ybus)

vecx_shunts = np.zeros((n, 1), dtype=complex)  # vector with shunt admittances
for i in range(df_top.shape[0]):  # passar per totes les files
    vecx_shunts[df_top.iloc[i, 0], 0] = vecx_shunts[df_top.iloc[i, 0], 0] + df_top.iloc[
        i, 4] * -1j  # B/2 is in column 4. The sign is changed here
    vecx_shunts[df_top.iloc[i, 1], 0] = vecx_shunts[df_top.iloc[i, 1], 0] + df_top.iloc[
        i, 4] * -1j  # B/2 is in column 4. The sign is changed here

vec_shunts = np.zeros((n - 1), dtype=complex)  # same vector, just to adapt
for i in range(n - 1):
    vec_shunts[i] = vecx_shunts[i + 1, 0]

vec_Y0 = np.zeros(n - 1, dtype=complex)  # vector with admittances connecting to the slack

for i in range(df_top.shape[0]):  # go through all rows

    if df_top.iloc[i, 0] == 0:  # if slack in the first column

        vec_Y0[df_top.iloc[i, 1] - 1] = 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)  # -1 so bus 1 goes to index 0

    elif df_top.iloc[i, 1] == 0:  # if slack in the second column

        vec_Y0[df_top.iloc[i, 0] - 1] = 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)

# --------------------------- INITIAL DATA: Y, SHUNTS AND Y0i. DONE


# --------------------------- INITIAL DATA: BUSES INFORMATION
df_bus = pd.read_excel('Data.xlsx', sheet_name='Busos')  # dataframe of the buses
if df_bus.shape[0] != n:
    print('Error: número de busos de ''Topologia'' i de ''Busos'' no és igual')  # check if number of buses is coherent


pq_list = list()  # vector to store the indices of PQ buses
pv_list = list()  # vector to store the indices of PV buses
vec_P = np.zeros(n - 1, dtype=float)  # data of active power
vec_Q = np.zeros(n - 1, dtype=float)  # data of reactive power
vec_V = np.zeros(n - 1, dtype=float)  # data of voltage magnitude
vec_W = np.zeros(n - 1, dtype=float)  # voltage magnitude squared

for i in range(df_bus.shape[0]):  # find the voltage specified for the slack
    if df_bus.iloc[i, 0] == 0:
        V_slack = df_bus.iloc[i, 3]
    else:
        V_slack = 1

Q0 = np.zeros(n)
V0 = np.zeros(n)
for i in range(df_bus.shape[0]):  # store the data of both PQ and PV
    vec_P[df_bus.iloc[i, 0] - 1] = df_bus.iloc[i, 1]  # -1 to start at 0
    if df_bus.iloc[i, 4] == 'PQ':
        Q0[i] = df_bus.iloc[i, 2]  # -1 to start at 0
        pq_list.append(i)
    elif df_bus.iloc[i, 4] == 'PV':
        V0[i] = df_bus.iloc[i, 3]  # -1 to start at 0
        pv_list.append(i)


pq = np.array(pq_list, dtype=int)
pv = np.array(pv_list, dtype=int)
pqpv = np.sort(np.r_[pq, pv])
npq = len(pq)
npv = len(pv)
npqpv = npq + npv

vec_Q = Q0[pqpv]
vec_V = V0[pqpv]
# --------------------------- INITIAL DATA: BUSES INFORMATION. DONE

# --------------------------- PREPARING IMPLEMENTATION
prof = 3  # depth
U = np.zeros((prof, npqpv), dtype=complex)  # voltages
U_re = np.zeros((prof, npqpv), dtype=float)  # real part of voltages
U_im = np.zeros((prof, npqpv), dtype=float)  # imaginary part of voltages
X = np.zeros((prof, npqpv), dtype=complex)  # X=1/conj(U)
X_re = np.zeros((prof, npqpv), dtype=float)  # real part of X
X_im = np.zeros((prof, npqpv), dtype=float)  # imaginary part of X
Q = np.zeros((prof, npqpv), dtype=complex)  # unknown reactive powers
vec_W = vec_V * vec_V
dimensions = 2 * npq + 3 * npv  # number of unknowns

Yred = Ybus[np.ix_(pqpv, pqpv)]  # admittance matrix without slack bus
G = np.real(Yred)  # real parts of Yij
B = np.imag(Yred)  # imaginary parts of Yij

# .......................CALCULATION OF TERMS [0]

U[0, :] = spsolve(Yred, vec_Y0)
X[0, :] = 1 / np.conj(U[0, :])
U_re[0, :] = U[0, :].real
U_im[0, :] = U[0, :].imag
X_re[0, :] = X[0, :].real
X_im[0, :] = X[0, :].imag
# .......................CALCULATION OF TERMS [0]. DONE

# .......................CALCULATION OF TERMS [1]
valor = np.zeros(npqpv, dtype=complex)
valor[pq - 1] = (V_slack - 1) * vec_Y0[pq - 1] + (vec_P[pq - 1] - vec_Q[pq - 1] * 1j) * X[0, pq - 1] + U[0, pq - 1] * vec_shunts[pq - 1]
valor[pv - 1] = (V_slack - 1) * vec_Y0[pv - 1] + (vec_P[pv - 1]) * X[0, pv - 1] + U[0, pv - 1] * vec_shunts[pv - 1]

RHS = np.zeros(2*(npqpv) + npv, dtype=float)
RHS[pq - 1] = valor[pq - 1].real
RHS[pv - 1] = valor[pv - 1].real
RHS[npqpv + (pq - 1)] = valor[pq - 1].imag
RHS[npqpv + (pv - 1)] = valor[pv - 1].imag
RHS[2 * (npqpv):] = vec_W[pv - 1] - 1


MAT = lil_matrix((dimensions, dimensions), dtype=float)
MAT[:(npqpv), :(npqpv)] = G
MAT[(npqpv):2 * (n - 1), :(n - 1)] = B
MAT[:(n - 1), (n - 1):2 * (n - 1)] = -B
MAT[(n - 1):2 * (n - 1), (n - 1):2 * (n - 1)] = G

MAT_URE = np.zeros((n - 1, n - 1), dtype=float)
np.fill_diagonal(MAT_URE, 2 * U_re[0, :])
MAT[2 * (n - 1):, :(n - 1)] = np.delete(MAT_URE, list(pq - 1), axis=0)

MAT_UIM = np.zeros((n - 1, n - 1), dtype=float)
np.fill_diagonal(MAT_UIM, 2 * U_im[0, :])
MAT[2 * (n - 1):, (n - 1):2 * (n - 1)] = np.delete(MAT_UIM, list(pq - 1), axis=0)

MAT_XIM = np.zeros((n - 1, n - 1), dtype=float)
np.fill_diagonal(MAT_XIM, -X_im[0, :])
MAT[:(n - 1), 2 * (n - 1):] = np.delete(MAT_XIM, list(pq - 1), axis=1)

MAT_XRE = np.zeros((n - 1, n - 1), dtype=float)
np.fill_diagonal(MAT_XRE, X_re[0, :])
MAT[(n-1):2 * (n - 1), 2 * (n - 1):] = np.delete(MAT_XRE, list(pq - 1), axis=1)

# factorize (only once)
MAT_csc = factorized(MAT.tocsc())

# solve
LHS = MAT_csc(RHS)

U_re[1, :] = LHS[:(n - 1)]
U_im[1, :] = LHS[(n - 1):2 * (n - 1)]
Q[0, pv - 1] = LHS[2 * (n - 1):]

U[1, :] = U_re[1, :] + U_im[1, :] * 1j
X[1, :] = (-X[0, :] * np.conj(U[1, :])) / np.conj(U[0, :])
X_re[1, :] = X[1, :].real
X_im[1, :] = X[1, :].imag

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

    valor[pq - 1] = (vec_P[pq - 1] - vec_Q[pq - 1] * 1j) * X[c - 1, pq - 1] + U[c - 1, pq - 1] * vec_shunts[pq - 1]
    valor[pv - 1] = conv(X, Q, c, pv - 1, 2) * -1j + U[c - 1, pv - 1] * vec_shunts[pv - 1] + X[c - 1, pv - 1] * vec_P[pv - 1]

    RHS[pq - 1] = valor[pq - 1].real
    RHS[pv - 1] = valor[pv - 1].real
    RHS[n - 1 + (pq - 1)] = valor[pq - 1].imag
    RHS[n - 1 + (pv - 1)] = valor[pv - 1].imag
    RHS[2 * (n - 1):] = -conv(U, U, c, pv - 1, 3).real  # the convolution of 2 complex is real :)

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
I_serie = Yred * U_final  # current flowing through series elements
I_inj_slack = vec_Y0 * V_slack
I_shunt = np.zeros((n - 1), dtype=complex)  # current through shunts
I_shunt[:] = -U_final * vec_shunts  # change the sign again
I_generada = I_serie - I_inj_slack + I_shunt  # current leaving the bus

# assembly the reactive power vector
Qfinal = vec_Q.copy()
Qfinal[pv-1] = (Q[:, pv-1] * 1j).sum(axis=0).imag

# compute the current injections
I_gen2 = (vec_P - vec_Q * 1j) / np.conj(U_final)

# print(U_final)
# print(abs(U_final))
# print(I_gen2 - I_generada)  # current balance. Should be almost 0
Qdf = pd.DataFrame(Q)  # to check the unknown reactive power
Qdf.to_excel('Results_reactive_power.xlsx', index=False, header=False)
Udf = pd.DataFrame(U)
Udf.to_excel('Results_voltage_coefficients.xlsx', index=False, header=False)  # to check the voltages

df = pd.DataFrame(np.c_[np.abs(U_final), np.angle(U_final), Qfinal, np.abs(I_gen2 - I_generada)],
                  columns=['|V|', 'Angle', 'Q', 'I error'])
print(df)
# test

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