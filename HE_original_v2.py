
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


# --------------------------- INITIAL DATA: Y, SHUNTS AND Y0i
df_top = pd.read_excel('Datax.xlsx', sheet_name='Topologia')  # DataFrame of the topology
df_bus = pd.read_excel('Datax.xlsx', sheet_name='Busos')  # Dataframe of the buses

n = df_bus.shape[0]  # number of buses, including slacks
nl = df_top.shape[0]  # number of lines

A = np.zeros((n, nl), dtype=int)  # incidence matrix
L = np.zeros((nl, nl), dtype=complex)  # matriu que conté les branques sèrie
np.fill_diagonal(L, [1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) for i in range(nl)])
A[df_top.iloc[range(nl), 0], range(nl)] = 1  # buses names must be >= 0 and integers
A[df_top.iloc[range(nl), 1], range(nl)] = -1

Yseries = np.dot(np.dot(A, L), np.transpose(A))
Yseries_slack = np.zeros((n,n), dtype=complex)
Yseries_slack[:,:] = Yseries[:,:]

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
        sl.append(i)
        V_sl.append(df_bus.iloc[i, 3]*(np.cos(df_bus.iloc[i, 4])+np.sin(df_bus.iloc[i, 4])*1j))

pq = np.array(pq)
pv = np.array(pv)
sl = np.array(sl)
npq = len(pq)
npv = len(pv)
nsl = len(sl)
npqpv = npq + npv

pqpv_x = np.sort(np.r_[pq, pv])  # ordeno els vectors amb incògnites
pqpv=[]
[pqpv.append(int(pqpv_x[i])) for i in range(len(pqpv_x))]  # per convertir els índexs a enters

pq_x = pq  # amb índexs originals
pv_x = pv  # amb índexs originals

vec_P = vec_Pi[pqpv]  # these 3 vectors are needed during the implementation
vec_Q = vec_Qi[pqpv]
vec_V = vec_Vi[pqpv]

Yshunts = np.zeros(n, dtype=complex)

for i in range(nl):  # de la sheet topologia
    Yshunts[df_top.iloc[i, 0]] += df_top.iloc[i, 4] * 1j  # no canvio el signe. Ysh = jB/2
    Yshunts[df_top.iloc[i, 1]] += df_top.iloc[i, 4] * 1j  # no canvio el signe. Ysh = jB/2
for i in range(n):  # de la sheet busos
    Yshunts[df_bus.iloc[i, 0]] += df_bus.iloc[i, 6] * 1j  # no canvio el signe

Yshunts_slack = np.zeros(n, dtype=complex)
Yshunts_slack[:] = Yshunts[:]

df = pd.DataFrame(data=np.c_[Yshunts.imag, vec_Pi, vec_Qi, vec_Vi],
                  columns=['Ysh', 'P0', 'Q0', 'V0'])

print(df)

Yslack = np.zeros((n, n), dtype=complex)
for i in range(nl):  # go through all rows
    if df_top.iloc[i, 0] in sl:  # if slack in the first column
        Yslack[df_top.iloc[i, 1], df_top.iloc[i, 0]] += 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)
    elif df_top.iloc[i, 1] in sl:  # if slack in the second column
        Yslack[df_top.iloc[i, 0], df_top.iloc[i, 1]] += 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)

Yslack = Yslack[:, sl]


# --------------------------- INITIAL DATA: BUSES INFORMATION. DONE

# --------------------------- PREPARING IMPLEMENTATION
prof = 30  # depth

U = np.zeros((prof, npqpv), dtype=complex)  # voltages. Deixo de mirar l'slack
U_re = np.zeros((prof, npqpv), dtype=float)  # real part of voltages
U_im = np.zeros((prof, npqpv), dtype=float)  # imaginary part of voltages
X = np.zeros((prof, npqpv), dtype=complex)  # X=1/conj(U)
X_re = np.zeros((prof, npqpv), dtype=float)  # real part of X
X_im = np.zeros((prof, npqpv), dtype=float)  # imaginary part of X
Q = np.zeros((prof, npqpv), dtype=complex)  # unknown reactive powers

W = vec_V * vec_V
dim = 2 * npq + 3 * npv  # number of unknowns
Yseries = Yseries[np.ix_(pqpv, pqpv)]  # combino tots els busos pq i pv per a reduir-la
G = np.real(Yseries)  # real parts of Yij
B = np.imag(Yseries)  # imaginary parts of Yij
Yshunts = Yshunts[pqpv]  # la redueixo
Yslack = Yslack[pqpv, :]  # la redueixo

# indices 0 based in the reduced scheme
nsl_counted = np.zeros(n, dtype=int)
compt = 0
for i in range(n):
    if i in sl:
        compt += 1
    nsl_counted[i] = compt

if npq > 0:
    pq_ = pq - nsl_counted[pq]
else:
    pq_ = []
if npv > 0:
    pv_ = pv - nsl_counted[pv]
else:
    pv_ = []

pqpv_x = np.sort(np.r_[pq_, pv_])
pqpv_ = []
[pqpv_.append(int(pqpv_x[i])) for i in range(len(pqpv_x))]  # per convertir els índexs a enters

print(pqpv_)

# .......................CALCULATION OF TERMS [0].......................
U_re[0, pqpv_] = 1  # pot ser qualsevol valor però forcem l'1 ja que V_sl[0] = 1, estat de referència
U_im[0, pqpv_] = 0  # pot ser qualsevol valor però forcem l'1 ja que V_sl[0] = 1, estat de referència
U[0, pqpv_] = U_re[0, pqpv_] + U_im[0, pqpv_] * 1j

X[0, pqpv_] = 1 / np.conj(U[0, pqpv_])
X_re[0, pqpv_] = np.real(X[0, pqpv_])
X_im[0, pqpv_] = np.imag(X[0, pqpv_])

Q[0, pv_] = 0  # estat de referència
# .......................CALCULATION OF TERMS [0]. DONE .......................

# .......................CALCULATION OF TERMS [1] .......................
valor = np.zeros(npqpv, dtype=complex)
prod = np.dot((Yslack[pqpv_, :]), V_sl[:])

valor[pq_] = prod[pq_] \
             - np.sum(Yslack[pq_, :], axis=1) \
             - Yshunts[pq_] * U[0, pq_] \
             + (vec_P[pq_] - vec_Q[pq_] * 1j) * X[0, pq_]

valor[pv_] = prod[pv_] \
             - np.sum(Yslack[pv_, :], axis=1) \
             - Yshunts[pv_] * U[0, pv_] \
             + vec_P[pv_] * X[0, pv_]

RHS = np.r_[valor.real, valor.imag, W[pv_] - 1]

print('RHS',RHS)

VRE = coo_matrix((2 * U_re[0, pv_], (np.arange(npv), pv_)), shape=(npv, npqpv)).tocsc()  # cada element ha de ser 2
VIM = coo_matrix((2 * U_im[0, pv_], (np.arange(npv), pv_)), shape=(npv, npqpv)).tocsc()  # cada element ha de ser 0
XIM = coo_matrix((-X_im[0, pv_], (pv_, np.arange(npv))), shape=(npqpv, npv)).tocsc()  # cada element ha de ser 0
XRE = coo_matrix((X_re[0, pv_], (pv_, np.arange(npv))), shape=(npqpv, npv)).tocsc()  # cada element ha de ser 1
EMPTY = csc_matrix((npv, npv))

MAT = vstack((hstack((G, -B, XIM)),
              hstack((B, G, XRE)),
              hstack((VRE, VIM, EMPTY))), format='csc')

print(MAT)
MAT_LU = factorized(MAT.tocsc())
LHS = MAT_LU(RHS)

U_re[1, :] = LHS[:npqpv]
U_im[1, :] = LHS[npqpv:2 * npqpv]
Q[1, pv_] = LHS[2 * npqpv:]

U[1, :] = U_re[1, :] + U_im[1, :] * 1j
X[1, :] = (-X[0, :] * np.conj(U[1, :])) / np.conj(U[0, :])
X_re[1, :] = X[1, :].real
X_im[1, :] = X[1, :].imag

print('LHS',LHS)

# .......................CALCULATION OF TERMS [1]. DONE .......................

# .......................CONVOLUCIONS .......................
def convQX(Q, X, i, c):
    suma = 0
    for k in range(c):
        suma += Q[k, i] * X[c-k, i]
    return suma

def convV(U, i, c):
    suma = 0
    for k in range(1, c):
        suma += U[k, i] * np.conj(U[c-k, i])
    return np.real(suma)

def convX(U, X, i, c):
    suma = 0
    for k in range(1, c+1):
        suma += np.conj(U[k, i]) * X[c-k, i]
    return suma

# .......................CONVOLUCIONS. DONE .......................

# .......................CALCULATION OF TERMS [c>=2] ........................
range_pqpv = np.arange(npqpv)

for c in range(2, prof):

    valor[pq_] = - Yshunts[pq_] * U[c-1, pq_] \
                 + (vec_P[pq_] - vec_Q[pq_] * 1j) * X[c-1, pq_]

    valor[pv_] = - Yshunts[pv_] * U[c-1, pv_] \
                 + vec_P[pv_] * X[c-1, pv_] \
                 - 1j * convQX(Q, X, pv_, c)

    RHS = np.r_[valor.real, valor.imag, -convV(U, pv_, c)]

    LHS = MAT_LU(RHS)

    U_re[c, :] = LHS[:npqpv]
    U_im[c, :] = LHS[npqpv:2 * npqpv]
    Q[c, pv_] = LHS[2 * npqpv:]

    U[c, :] = U_re[c, :] + U_im[c, :] * 1j
    X[c, :] = - convX(U, X, range_pqpv, c) / np.conj(U[0, :])
    X_re[c, :] = X[c, :].real
    X_im[c, :] = X[c, :].imag

# .......................CALCULATION OF TERMS [c>=2]. DONE ........................

# .......................RESULTATS ........................
Ufi = np.zeros(n, dtype=complex)
Pfi = np.zeros(n, dtype=complex)
Qfi = np.zeros(n, dtype=complex)

Ybus = Yseries_slack + diags(Yshunts_slack)

from Funcions import pade4all, epsilon, eta, theta, aitken, Sigma_funcO

#PADÉ
Upa = pade4all(prof-1, U[:, :], 1)
Qpa = pade4all(prof-1, Q[:, pv_], 1)
Ufi[sl] = V_sl
Ufi[pqpv] = Upa
Pfi[pqpv] = vec_P[pqpv_]
if npq > 0:
    Qfi[pq] = vec_Q[pq_]
if npv > 0:
    Qfi[pv] = Qpa
Pfi[sl] = np.nan
Qfi[sl] = np.nan
#FI PADÉ

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



df = pd.DataFrame(np.c_[np.abs(Ufi), np.angle(Ufi) * 180 / np.pi, np.abs(Pfi), np.abs(Qfi)],
                        columns=['|V|', 'Angle (deg)', 'P', 'Q'])

print(df)

#ERRORS
S_out = np.asarray(Ufi) * np.conj(np.asarray(np.dot(Ybus, Ufi)))
S_in = (Pfi[:] + 1j * Qfi[:])
error = S_in - S_out  # mismatch de potències
err = max(abs(np.r_[error[0, pqpv]]))  # màxim error de potències
print('Error màxim: ' + str(err))
#FI ERRORS

