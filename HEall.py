# AUTHOR: Josep Fanals Batllori
# CONTACT: u1946589@campus.udg.edu.

# --------------------------- LIBRARIES
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, factorized
np.set_printoptions(linewidth=2000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# --------------------------- END LIBRARIES

# --------------------------- INITIAL DATA: Y, SHUNTS AND Y0i
df_top = pd.read_excel('Dades_v1.xlsx', sheet_name='Topology')  # DataFrame of the topology

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
df_bus = pd.read_excel('Data.xlsx', sheet_name='Buses')  # dataframe of the buses
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
U_re = np.zeros((prof, n - 1), dtype=float)  # real part of voltages
U_im = np.zeros((prof, n - 1), dtype=float)  # imaginary part of voltages
X = np.zeros((prof, n - 1), dtype=complex)  # X=1/conj(U)
X_re = np.zeros((prof, n - 1), dtype=float)  # real part of X
X_im = np.zeros((prof, n - 1), dtype=float)  # imaginary part of X
Q = np.zeros((prof, n - 1), dtype=complex)  # unknown reactive powers
pqpv = np.r_[vec_busos_PQ, vec_busos_PV]
pq = vec_busos_PQ
pv = vec_busos_PV
np.sort(pqpv)
npq = len(pq)
npv = len(pv)
dimensions = 2 * npq + 3 * npv  # number of unknowns

# .......................GUIDING VECTOR
lx = 0
index_Ure = []
index_Uim = []
index_Q = []
for i in range(n-1):
    index_Ure.append(lx)
    index_Uim.append(lx + 1)
    if i + 1 in pq:
        lx = lx + 2
    else:
        index_Q.append(lx + 2)
        lx = lx + 3
# .......................GUIDING VECTOR. DONE

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

RHSx = np.zeros((3, n - 1), dtype=float)
RHSx[0, pq - 1] = valor[pq - 1].real
RHSx[1, pq - 1] = valor[pq - 1].imag
RHSx[2, pq - 1] = np.nan  # to later delete
RHSx[0, pv - 1] = valor[pv - 1].real
RHSx[1, pv - 1] = valor[pv - 1].imag
RHSx[2, pv - 1] = vec_W[pv - 1, 0] - 1

rhs = np.matrix.flatten(RHSx, 'f')
rhs = rhs[~np.isnan(rhs)]  # delete dummy cells

mat = np.zeros((dimensions, dimensions), dtype=float)  # constant matrix

k = 0  # index that will go through the rows
for i in range(n - 1):  # fill the matrix
    lx = 0
    for j in range(n - 1):
        mat[k, lx] = G[i, j]
        mat[k + 1, lx] = B[i, j]
        mat[k, lx + 1] = -B[i, j]
        mat[k + 1, lx + 1] = G[i, j]
        if (j == i) and (i + 1 in pv) and (j + 1 in pv):
            mat[k + 2, lx] = 2 * U_re[0, i]
            mat[k + 2, lx + 1] = 2 * U_im[0, i]
            mat[k, lx + 2] = -X_im[0, i]
            mat[k + 1, lx + 2] = X_re[0, i]
        lx = lx + 2 if (j + 1 in pq) else lx + 3
    k = k + 2 if (i + 1 in pq) else k + 3

mat_factorized = factorized(csc_matrix(mat))
lhs = mat_factorized(rhs)

U_re[1, :] = lhs[index_Ure]
U_im[1, :] = lhs[index_Uim]
Q[0, pv - 1] = lhs[index_Q]

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
        return sum(suma).real


for c in range(2, prof):  # c defines the current depth
    valor[pq - 1] = (vec_P[pq - 1, 0] - vec_Q[pq - 1, 0] * 1j) * X[c - 1, pq - 1] + U[c - 1, pq - 1] * vec_shunts[pq - 1, 0]
    valor[pv - 1] = conv(X, Q, c, pv - 1, 2) * -1j + U[c - 1, pv - 1] * vec_shunts[pv - 1, 0] + X[c - 1, pv - 1] * vec_P[pv - 1, 0]

    RHSx[0, pq - 1] = valor[pq - 1].real
    RHSx[1, pq - 1] = valor[pq - 1].imag
    RHSx[2, pq - 1] = np.nan  # per poder-ho eliminar bé, dummy

    RHSx[0, pv - 1] = valor[pv - 1].real
    RHSx[1, pv - 1] = valor[pv - 1].imag
    RHSx[2, pv - 1] = -conv(U, U, c, pv - 1, 3).real  # square of the voltage stores the value at the real part

    rhs = np.matrix.flatten(RHSx, 'f')
    rhs = rhs[~np.isnan(rhs)]

    lhs = mat_factorized(rhs)

    U_re[c, :] = lhs[index_Ure]
    U_im[c, :] = lhs[index_Uim]
    Q[c - 1, pv - 1] = lhs[index_Q]

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
I_gen2 = np.array(I_gen2)


print(I_gen2 - I_generada)  # current balance. Should be almost 0
Qdf = pd.DataFrame(Q)  # to check the unknown reactive power
Qdf.to_excel('Results_reactive_power.xlsx', index=False, header=False)
Udf = pd.DataFrame(U)
Udf.to_excel('Results_voltage_coefficients.xlsx', index=False, header=False)  # to check the voltages

print('Coefficients')
print(Udf)

# ----------------------------------------------------------------------------------------------------------------------
# Newton-Raphson comparison
# ----------------------------------------------------------------------------------------------------------------------
V_nr = np.array([1.+0.j,
                 0.9534313 -0.02268264j,
                 0.94114008-0.03194917j,
                 0.93894126-0.01876796j,
                 0.94938523-0.03417151j,
                 0.93991439-0.0126862j ,
                 0.92929251-0.02953734j,
                 0.93540855-0.02732348j,
                 0.9097992 -0.01911601j,
                 0.94537686-0.03929395j,
                 0.97988783-0.01482682j,
                 0.91993789-0.01068987j])  # voltage from GridCal with 1e-7 error

ok = np.isclose(np.abs(U_final), np.abs(V_nr[1:]), atol=1e-2).all()
print('Test passed', ok)
if not ok:
    print('It should be:')
    print(np.abs(V_nr[1:]))

