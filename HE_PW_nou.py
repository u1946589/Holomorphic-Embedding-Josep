
# AUTHORS: Santiago Peñate Vera and Josep Fanals Batllori
# CONTACT:  santiago.penate.vera@gmail.com, u1946589@campus.udg.edu
# thanks to Llorenç Fanals Batllori for his help at coding
# netejo el codi, seguint els PEP. Comentaris en català

# --------------------------- LLIBRERIES
import numpy as np
import numba as nb
from numba import jit
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
pd.set_option("display.precision", 5)
# --------------------------- FI LLIBRERIES

# --------------------------- CÀRREGA DE DADES INICIALS
df_top = pd.read_excel('IEEE30.xlsx', sheet_name='Topologia')  # dades de la topologia
df_bus = pd.read_excel('IEEE30.xlsx', sheet_name='Busos')  # dades dels busos

n = df_bus.shape[0]  # nombre de busos, inclou l'slack
nl = df_top.shape[0]  # nombre de línies

A = np.zeros((n, nl), dtype=int)  # matriu d'incidència, formada per 1, -1 i 0
L = np.zeros((nl, nl), dtype=complex)  # matriu amb les branques
np.fill_diagonal(L, [1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) for i in range(nl)])
A[df_top.iloc[range(nl), 0], range(nl)] = 1
A[df_top.iloc[range(nl), 1], range(nl)] = -1

Yseries = np.dot(np.dot(A, L), np.transpose(A))  # matriu de les branques sèrie
Yseries_slack = np.zeros((n, n), dtype=complex)
Yseries_slack[:, :] = Yseries[:, :]  # també conté les admitàncies amb l'slack

Ytap = np.zeros((n, n), dtype=complex)  # diferència entre Ytapreal i Yseries (aquesta última conté Ys simètrica)

for i in range(nl):  # emplenar matriu quan hi ha trafo de relació variable
    if df_top.iloc[i, 5] != 1:
        Ytap[df_top.iloc[i, 0], df_top.iloc[i, 0]] += 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) / \
            (df_top.iloc[i, 5] * np.conj(df_top.iloc[i, 5])) - 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)
        Ytap[df_top.iloc[i, 1], df_top.iloc[i, 1]] += 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) \
            - 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)
        Ytap[df_top.iloc[i, 0], df_top.iloc[i, 1]] += - 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) / \
            (np.conj(df_top.iloc[i, 5])) + 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)
        Ytap[df_top.iloc[i, 1], df_top.iloc[i, 0]] += - 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) / \
            (df_top.iloc[i, 5]) + 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)

vec_Pi = np.zeros(n, dtype=float)  # dades de potència activa
vec_Qi = np.zeros(n, dtype=float)  # dades de potència reactiva
vec_Vi = np.zeros(n, dtype=float)  # dades de tensió
vec_Wi = np.zeros(n, dtype=float)  # tensió al quadrat

pq = []  # índexs dels busos PQ
pv = []  # índexs dels busos PV
sl = []  # índexs dels busos slack
vec_Pi[:] = np.nan_to_num(df_bus.iloc[:, 1])  # emplenar el vector de números
vec_Qi[:] = np.nan_to_num(df_bus.iloc[:, 2])
vec_Vi[:] = np.nan_to_num(df_bus.iloc[:, 3])
V_sl = []  # tensions slack

for i in range(n):  # cerca per a guardar els índexs
    if df_bus.iloc[i, 5] == 'PQ':
        pq.append(i)
    elif df_bus.iloc[i, 5] == 'PV':
        pv.append(i)
    elif df_bus.iloc[i, 5] == 'Slack':
        sl.append(i)
        V_sl.append(df_bus.iloc[i, 3]*(np.cos(df_bus.iloc[i, 4])+np.sin(df_bus.iloc[i, 4])*1j))

pq = np.array(pq)  # índexs en forma de vector
pv = np.array(pv)
sl = np.array(sl)
npq = len(pq)  # nombre de busos PQ
npv = len(pv)
nsl = len(sl)
npqpv = npq + npv  # nombre de busos incògnita

pqpv_x = np.sort(np.r_[pq, pv])  # ordenar els vectors amb incògnites
pqpv = []
[pqpv.append(int(pqpv_x[i])) for i in range(len(pqpv_x))]  # convertir els índexs a enters

pq_x = pq  # guardar els índexs originals
pv_x = pv

vec_P = vec_Pi[pqpv]  # agafar la part del vector necessària
vec_Q = vec_Qi[pqpv]
vec_V = vec_Vi[pqpv]

Yshunts = np.zeros(n, dtype=complex)

for i in range(nl):  # de la pestanya topologia
    Yshunts[df_top.iloc[i, 0]] += df_top.iloc[i, 4] * 1j  # es donen en forma d'admitàncies
    Yshunts[df_top.iloc[i, 1]] += df_top.iloc[i, 4] * 1j
for i in range(n):  # de la pestanya busos
    Yshunts[df_bus.iloc[i, 0]] += df_bus.iloc[i, 6] * 1j

Yshunts_slack = np.zeros(n, dtype=complex)  # inclou els busos slack
Yshunts_slack[:] = Yshunts[:]

df = pd.DataFrame(data=np.c_[Yshunts.imag, vec_Pi, vec_Qi, vec_Vi],
                  columns=['Ysh', 'P0', 'Q0', 'V0'])
print(df)

Yslack = Yseries_slack[:, sl]  # les columnes pertanyents als slack
# --------------------------- FI CÀRREGA DE DADES INICIALS

# --------------------------- PREPARACIÓ DE LA IMPLEMENTACIÓ
prof = 30  # nombre de coeficients de les sèries

U = np.zeros((prof, npqpv), dtype=complex)  # sèries de voltatges
U_re = np.zeros((prof, npqpv), dtype=float)  # part real de voltatges
U_im = np.zeros((prof, npqpv), dtype=float)  # part imaginària de voltatges
X = np.zeros((prof, npqpv), dtype=complex)  # inversa de la tensió conjugada
X_re = np.zeros((prof, npqpv), dtype=float)  # part real d'X
X_im = np.zeros((prof, npqpv), dtype=float)  # part imaginària d'X
Q = np.zeros((prof, npqpv), dtype=complex)  # sèries de potències reactives

W = vec_V * vec_V  # mòdul de les tensions al quadrat
dim = 2 * npq + 3 * npv  # nombre d'incògnites
Yseries = Yseries[np.ix_(pqpv, pqpv)]  # reduir per a deixar de banda els slack
Ytaps = Ytap[np.ix_(pqpv, pqpv)]  # reduir per a deixar de banda els slack
Ytapslack = Ytap[np.ix_(pqpv, sl)]  # columnes de la matriu d'admitàncies asimètrica per als slack
G = np.real(Yseries)  # part real de la matriu simètrica
B = np.imag(Yseries)  # part imaginària de la matriu simètrica
Yshunts = Yshunts[pqpv]  # reduir per a deixar de banda els slack
Yslack = Yslack[pqpv, :]  # files que enllacen amb els busos PQ i PV

nsl_counted = np.zeros(n, dtype=int)  # nombre de busos slack trobats abans d'un bus
compt = 0
for i in range(n):  # per a índexs que comencin del 0
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
if nsl > 0:
    sl_ = sl - nsl_counted[sl]

pqpv_x = np.sort(np.r_[pq_, pv_])  # ordenar els nous índexs dels busos PQ i PV
pqpv_ = []
[pqpv_.append(int(pqpv_x[i])) for i in range(len(pqpv_x))]  # convertir els índexs a enters
# --------------------------- FI PREPARACIÓ DE LA IMPLEMENTACIÓ

# .......................TERMES [0].......................
U_re[0, pqpv_] = 1  # estat de referència
U_im[0, pqpv_] = 0
U[0, pqpv_] = U_re[0, pqpv_] + U_im[0, pqpv_] * 1j
X[0, pqpv_] = 1 / np.conj(U[0, pqpv_])
X_re[0, pqpv_] = np.real(X[0, pqpv_])
X_im[0, pqpv_] = np.imag(X[0, pqpv_])
Q[0, pv_] = 0  # estat de referència
# .......................FI TERMES [0] .......................

# ....................... TERMES [1] .......................
range_pqpv = np.arange(npqpv)  # tots els busos ordenats
valor = np.zeros(npqpv, dtype=complex)  # vector auxiliar per a guardar parts del RHS

prod = np.dot((Yslack[pqpv_, :]), V_sl[:])  # intensitat que injecten els slack
prod2 = np.dot((Ytaps[pqpv_, :]), U[0, :])  # itensitat retardada que considera la matriu asimètrica

valor[pq_] = - prod[pq_] \
             + np.sum(Yslack[pq_, :], axis=1) \
             - Yshunts[pq_] * U[0, pq_] \
             + (vec_P[pq_] - vec_Q[pq_] * 1j) * X[0, pq_] \
             - prod2[pq_] \
             - np.sum(Ytapslack[pq_, :], axis=1)

valor[pv_] = - prod[pv_] \
             + np.sum(Yslack[pv_, :], axis=1) \
             - Yshunts[pv_] * U[0, pv_] \
             + vec_P[pv_] * X[0, pv_] \
             - prod2[pv_] \
             - np.sum(Ytapslack[pv_, :], axis=1)

RHS = np.r_[valor.real, valor.imag, W[pv_] - 1]  # amb l'equació del mòdul dels PV

VRE = coo_matrix((2 * U_re[0, pv_], (np.arange(npv), pv_)), shape=(npv, npqpv)).tocsc()  # matriu dispersa COO a compr.
VIM = coo_matrix((2 * U_im[0, pv_], (np.arange(npv), pv_)), shape=(npv, npqpv)).tocsc()
XIM = coo_matrix((-X_im[0, pv_], (pv_, np.arange(npv))), shape=(npqpv, npv)).tocsc()
XRE = coo_matrix((X_re[0, pv_], (pv_, np.arange(npv))), shape=(npqpv, npv)).tocsc()
EMPTY = csc_matrix((npv, npv))  # matriu dispera comprimida

MATx = vstack((hstack((G, -B, XIM)),
              hstack((B, G, XRE)),
              hstack((VRE, VIM, EMPTY))), format='csc')

MAT_LU = factorized(MATx.tocsc())  # matriu factoritzada (només cal fer-ho una vegada)
LHS = MAT_LU(RHS)  # obtenir vector d'incògnites

U_re[1, :] = LHS[:npqpv]  # part real de les tensions
U_im[1, :] = LHS[npqpv:2 * npqpv]  # part imaginària de les tensions
Q[1, pv_] = LHS[2 * npqpv:]  # potència reactiva

U[1, :] = U_re[1, :] + U_im[1, :] * 1j
X[1, :] = (-X[0, :] * np.conj(U[1, :])) / np.conj(U[0, :])
X_re[1, :] = X[1, :].real
X_im[1, :] = X[1, :].imag
# .......................FI TERMES [1] .......................

# .......................CONVOLUCIONS .......................


def convqx(q, x, i, cc):
    suma = 0
    for k in range(cc):
        suma += q[k, i] * x[cc-k, i]
    return suma


def convv(u, i, cc):
    suma = 0
    for k in range(1, cc):
        suma += u[k, i] * np.conj(u[cc-k, i])
    return np.real(suma)


def convx(u, x, i, cc):
    suma = 0
    for k in range(1, cc+1):
        suma += np.conj(u[k, i]) * x[cc-k, i]
    return suma
# .......................FI CONVOLUCIONS .......................

# .......................TERMES [2] ........................
prod2 = np.dot((Ytaps[pqpv_, :]), U[1, :])  # càlcul amb la matriu asimètrica retardat
prod3 = np.dot((Ytapslack[pqpv_, :]), V_sl[:])  # càlcul amb la matriu asimètrica retardada pels slack
c = 2  # profunditat actual

valor[pq_] = - Yshunts[pq_] * U[c-1, pq_] \
             + (vec_P[pq_] - vec_Q[pq_] * 1j) * X[c-1, pq_] \
             - prod2[pq_] \
             - np.sum(Ytapslack[pq_, :], axis=1) * (-1) \
             - prod3[pq_]

valor[pv_] = - Yshunts[pv_] * U[c-1, pv_] \
             + vec_P[pv_] * X[c-1, pv_] \
             - 1j * convqx(Q, X, pv_, c) \
             - prod2[pv_] \
             - np.sum(Ytapslack[pv_, :], axis=1) * (-1) \
             - prod3[pv_]

RHS = np.r_[valor.real, valor.imag, -convv(U, pv_, c)]

LHS = MAT_LU(RHS)

U_re[c, :] = LHS[:npqpv]
U_im[c, :] = LHS[npqpv:2 * npqpv]
Q[c, pv_] = LHS[2 * npqpv:]

U[c, :] = U_re[c, :] + U_im[c, :] * 1j
X[c, :] = - convx(U, X, range_pqpv, c) / np.conj(U[0, :])
X_re[c, :] = X[c, :].real
X_im[c, :] = X[c, :].imag
# .......................FI TERMES [2] ........................

# .......................TERMES [c>=3] ........................
for c in range(3, prof):
    prod2 = np.dot((Ytaps[pqpv_, :]), U[c-1, :])

    valor[pq_] = - Yshunts[pq_] * U[c-1, pq_] \
        + (vec_P[pq_] - vec_Q[pq_] * 1j) * X[c-1, pq_] \
        - prod2[pq_]

    valor[pv_] = - Yshunts[pv_] * U[c-1, pv_] \
        + vec_P[pv_] * X[c-1, pv_] \
        - 1j * convqx(Q, X, pv_, c) \
        - prod2[pv_]

    RHS = np.r_[valor.real, valor.imag, -convv(U, pv_, c)]

    LHS = MAT_LU(RHS)

    U_re[c, :] = LHS[:npqpv]
    U_im[c, :] = LHS[npqpv:2 * npqpv]
    Q[c, pv_] = LHS[2 * npqpv:]

    U[c, :] = U_re[c, :] + U_im[c, :] * 1j
    X[c, :] = - convx(U, X, range_pqpv, c) / np.conj(U[0, :])
    X_re[c, :] = X[c, :].real
    X_im[c, :] = X[c, :].imag
# .......................FI TERMES [c>3] ........................

# .......................RESULTATS ........................
Pfi = np.zeros(n, dtype=complex)  # potència activa final
Qfi = np.zeros(n, dtype=complex)  # potència reactiva final
U_sum = np.zeros(n, dtype=complex)  # tensió a partir la suma de coeficients
U_pa = np.zeros(n, dtype=complex)  # tensió amb Padé
U_th = np.zeros(n, dtype=complex)  # tensió amb Thévenin
U_ait = np.zeros(n, dtype=complex)  # tensió amb deltes quadrades d'Aitken
U_eps = np.zeros(n, dtype=complex)  # tensió amb èpsilons de Wynn
U_rho = np.zeros(n, dtype=complex)  # tensió amb rhos
U_theta = np.zeros(n, dtype=complex)  # tensió amb thetas
U_eta = np.zeros(n, dtype=complex)  # tensió amb etas
Q_eps = np.zeros(n, dtype=complex)
Q_ait = np.zeros(n, dtype=complex)
Q_rho = np.zeros(n, dtype=complex)
Q_theta = np.zeros(n, dtype=complex)
Q_eta = np.zeros(n, dtype=complex)
Sig_re = np.zeros(n, dtype=complex)  # part real de sigma
Sig_im = np.zeros(n, dtype=complex)  # part imaginària de sigma

Ybus = Yseries_slack + diags(Yshunts_slack) + Ytap  # matriu d'admitàncies total

from Funcions import pade4all, epsilon2, eta, theta, aitken, Sigma_funcO, rho, thevenin_funcX2  # importar funcions

# SUMA
U_sum[pqpv] = np.sum(U[:, pqpv_], axis=0)
U_sum[sl] = V_sl
# FI SUMA

# PADÉ
Upa = pade4all(prof-1, U[:, :], 1)
Qpa = pade4all(prof-1, Q[:, pv_], 1)  # trobar reactiva amb Padé
U_pa[sl] = V_sl
U_pa[pqpv] = Upa
Pfi[pqpv] = vec_P[pqpv_]
if npq > 0:
    Qfi[pq] = vec_Q[pq_]
if npv > 0:
    Qfi[pv] = Qpa
Pfi[sl] = np.nan
Qfi[sl] = np.nan
# FI PADÉ

limit = 10  # límit per tal que els mètodes recurrents no treballin amb tots els coeficients
if limit > prof:
    limit = prof - 1

# SIGMA
Ux1 = np.copy(U)
Sig_re[pqpv] = np.real(Sigma_funcO(Ux1, X, prof-1, V_sl))
Sig_im[pqpv] = np.imag(Sigma_funcO(Ux1, X, prof-1, V_sl))
Sig_re[sl] = np.nan
Sig_im[sl] = np.nan
s_p = 1 / (2 * (abs(np.real(Sig_re) + np.real(Sig_im) * 1j) - np.real(Sig_re)))
s_n = - 1 / (2 * (abs(np.real(Sig_re) + np.real(Sig_im) * 1j) + np.real(Sig_re)))
# FI SIGMA

# THÉVENIN
Ux2 = np.copy(U)
for i in range(npqpv):
    U_th[i] = thevenin_funcX2(Ux2[:limit, i], X[:limit, i], 1)
U_th[pqpv] = U_th[pqpv_]
U_th[sl] = V_sl
# FI THÉVENIN

# DELTES D'AITKEN
Ux3 = np.copy(U)
Qx3 = np.copy(Q)
for i in range(npqpv):
    U_ait[i] = aitken(Ux3[:, i], limit)
    if i in pq_:
        Q_ait[i + nsl_counted[i]] = vec_Q[i]
    elif i in pv_:
        Q_ait[i + nsl_counted[i]] = aitken(Qx3[:, i], limit)
U_ait[pqpv] = U_ait[pqpv_]
U_ait[sl] = V_sl
Q_ait[sl] = np.nan
# FI DELTES D'AITKEN

# EPSILONS DE WYNN
Ux4 = np.copy(U)
Qx4 = np.copy(Q)
for i in range(npqpv):
    U_eps[i] = epsilon2(Ux4[:, i], limit)
    if i in pq_:
        Q_eps[i + nsl_counted[i]] = vec_Q[i]
    elif i in pv_:
        Q_eps[i + nsl_counted[i]] = epsilon2(Qx4[:, i], limit)
U_eps[pqpv] = U_eps[pqpv_]
U_eps[sl] = V_sl
Q_eps[sl] = np.nan
# FI EPSILONS DE WYNN

# RHO
Ux5 = np.copy(U)
Qx5 = np.copy(Q)
for i in range(npqpv):
    U_rho[i] = rho(Ux5[:, i], limit)
    if i in pq_:
        Q_rho[i + nsl_counted[i]] = vec_Q[i]
    elif i in pv_:
        Q_rho[i + nsl_counted[i]] = rho(Qx5[:, i], limit)
U_rho[pqpv] = U_rho[pqpv_]
U_rho[sl] = V_sl
Q_rho[sl] = np.nan
# FI RHO

# THETA
Ux6 = np.copy(U)
Qx6 = np.copy(Q)
for i in range(npqpv):
    U_theta[i] = theta(Ux6[:, i], limit)
    if i in pq_:
        Q_theta[i + nsl_counted[i]] = vec_Q[i]
    elif i in pv_:
        Q_theta[i + nsl_counted[i]] = theta(Qx6[:, i], limit)
U_theta[pqpv] = U_theta[pqpv_]
U_theta[sl] = V_sl
Q_theta[sl] = np.nan
# FI THETA

# ETA
Ux7 = np.copy(U)
Qx7 = np.copy(Q)
for i in range(npqpv):
    U_eta[i] = eta(Ux7[:, i], limit)
U_eta[pqpv] = U_eta[pqpv_]
U_eta[sl] = V_sl
Q_eta[:] = Qfi[:]
# FI ETA

# CÀLCUL DELS ERRORS
S_out = np.asarray(U_pa) * np.conj(np.asarray(np.dot(Ybus, U_pa)))  # computat amb tensions de Padé
S_in = (Pfi[:] + 1j * Qfi[:])
error = S_in - S_out  # error final de potències

S_out_eps = np.asarray(U_eps) * np.conj(np.asarray(np.dot(Ybus, U_eps)))
S_in = (Pfi[:] + 1j * Q_eps[:])
err_eps = S_in - S_out_eps
errm_eps = max(abs(np.r_[err_eps[0, pqpv]]))
S_out_ait = np.asarray(U_ait) * np.conj(np.asarray(np.dot(Ybus, U_ait)))
S_in = (Pfi[:] + 1j * Q_ait[:])
err_ait = S_in - S_out_ait
errm_ait = max(abs(np.r_[err_ait[0, pqpv]]))
S_out_rho = np.asarray(U_rho) * np.conj(np.asarray(np.dot(Ybus, U_rho)))
S_in = (Pfi[:] + 1j * Q_rho[:])
err_rho = S_in - S_out_rho
errm_rho = max(abs(np.r_[err_rho[0, pqpv]]))
S_out_theta = np.asarray(U_theta) * np.conj(np.asarray(np.dot(Ybus, U_theta)))
S_in = (Pfi[:] + 1j * Q_theta[:])
err_theta = S_in - S_out_theta
errm_theta = max(abs(np.r_[err_theta[0, pqpv]]))
S_out_eta = np.asarray(U_eta) * np.conj(np.asarray(np.dot(Ybus, U_eta)))
S_in = (Pfi[:] + 1j * Qfi[:])  # si no faig servir la de Padé, surten np.nan
err_eta = S_in - S_out_eta
errm_eta = max(abs(np.r_[err_eta[0, pqpv]]))

# FI CÀLCUL DELS ERRORS

df = pd.DataFrame(np.c_[np.abs(U_sum), np.angle(U_sum), np.abs(U_pa), np.angle(U_pa), np.abs(U_th),
                        np.abs(U_eps), np.angle(U_eps), np.abs(U_ait), np.angle(U_ait), np.abs(U_rho),
                        np.angle(U_rho), np.abs(U_theta), np.angle(U_theta), np.abs(U_eta), np.angle(U_eta),
                        np.real(Pfi), np.real(Qfi), np.abs(error[0, :]), np.real(Sig_re), np.real(Sig_im), s_p, s_n],
                        columns=['|V| sum', 'A. sum', '|V| Padé', 'A. Padé', '|V| Thévenin', '|V| Epsilon',
                                 'A. Epsilon', '|V| Aitken', 'A. Aitken', '|V| Rho', 'A. Rho', '|V| Theta', 'A. Theta',
                                 '|V| Eta', 'A. Eta', 'P', 'Q', 'S error', 'Sigma re', 'Sigma im', 's+', 's-'])
print(df)

err = max(abs(np.r_[error[0, pqpv]]))  # màxim error de potències
print('Error màxim amb Padé: ' + str(err))
# print(errm_eps)
# print(errm_ait)
# print(errm_rho)
# print(errm_theta)
# print(errm_eta)


# --------------------------- PADÉ-WEIERSTRASS (P-W)

def vector_s0(vec, s_0):  # per a calcular V(s_0)
    suma = 0
    for k in range(len(vec)):
        suma += vec[k] * s_0 ** k
    return suma


s0 = 0.8

prof_pw = prof  # nombre de coeficients de les sèries del P-W

Up = np.zeros((prof_pw, npqpv), dtype=complex)  # tensions prima incògnita
Up_re = np.zeros((prof_pw, npqpv), dtype=float)
Up_im = np.zeros((prof_pw, npqpv), dtype=float)
Xp = np.zeros((prof_pw, npqpv), dtype=complex)
Xp_re = np.zeros((prof_pw, npqpv), dtype=float)
Xp_im = np.zeros((prof_pw, npqpv), dtype=float)
Qp = np.zeros((prof_pw, npqpv), dtype=complex)
Upw = np.zeros((prof_pw, nsl), dtype=complex)  # tensions prima dels slack
Us0 = np.zeros(n, dtype=complex)  # totes les tensions V(s0)
Qs0 = np.zeros(n, dtype=complex)  # totes les Q(s0) dels busos PV

V_sl = np.asarray(V_sl)  # convertir a vector

Us0[sl] = 1 + s0 * V_sl[nsl_counted[sl]-1] - s0 * 1  # emplenar la tensió dels slack
Upw[0] = 1
Upw[1, nsl_counted[sl]-1] = (1-s0)*(V_sl[:]-1)/(1+s0*(V_sl[:]-1))  # la resta de coeficients són nuls

Us0[pqpv] = vector_s0((U[:, pqpv_]), s0)  # emplenar la tensió dels busos incògnita

if npv > 0:
    Qs0[pv] = vector_s0((Q[:, pv_]), s0)  # emplenar la reactiva dels busos PV

Yahat = np.copy(Ytap)  # asimètrica
Ybhat = np.copy(Yseries_slack)  # simètrica

for i in range(n):
    if i not in sl:  # per la fila de l'slack no cal fer-ho
        for j in range(n):
            Yahat[i, j] = Yahat[i, j] * Us0[j] * np.conj(Us0[i])
            Ybhat[i, j] = Ybhat[i, j] * Us0[j] * np.conj(Us0[i])

gamma = np.zeros(npqpv, dtype=complex)
if npq > 0:
    gamma[pq_] = s0 * (Pfi[pq] - Qfi[pq] * 1j)  # gamma pels busos PQ
if npv > 0:
    gamma[pv_] = s0 * Pfi[pv] - Qs0[pv] * 1j  # gamma pels busos PV

Ybtilde = np.copy(Ybhat)  # matriu simètrica evolucionada
if npq > 0:
    Ybtilde[pq, pq] += +s0 * Yshunts_slack[pq] * abs(Us0[pq]) ** 2 - gamma[pq_]
if npv > 0:
    Ybtilde[pv, pv] += +s0 * Yshunts_slack[pv] * abs(Us0[pv]) ** 2 - gamma[pv_]

Ybtilde[:, :] += s0 * Yahat[:, :]  # ajustament, part que no s'incrusta amb s'
Yahat[:, :] = (1 - s0) * Yahat[:, :]  # ajustament, part que no s'incrusta amb s'

# .......................TERMES [0] ........................
Up[0, :] = 1  # estat de referència
Qp[0, :] = 0

Up_re[0, :] = np.real(Up[0, :])
Up_im[0, :] = np.imag(Up[0, :])
Xp[0, :] = 1 / Up[0, :]
Xp_re[0, :] = np.real(Xp[0, :])
Xp_im[0, :] = np.imag(Xp[0, :])
# .......................FI TERMES [0] ........................

Yahatred = Yahat[np.ix_(pqpv, pqpv)]  # asimètrica sense slack
Yahatw = Yahat[np.ix_(pqpv, sl)]  # asimètrica de l'slack
Ybtildered = Ybtilde[np.ix_(pqpv, pqpv)]  # simètrica sense slack
Ybtildew = Ybtilde[np.ix_(pqpv, sl)]  # simètrica amb slack

# .......................TERMES [1] ........................
prod1 = np.dot(Ybtildew[pqpv_, :], Upw[1, :])  # producte de la simètrica amb l'slack
prod2 = np.dot(Yahatred[pqpv_, :], Up[0, :])  # producte de l'asimètrica amb la tensió incògnita
prod3 = np.dot(Yahatw[pqpv_, :], Upw[0, :])  # producte de l'asimètrica amb l'slack

if npq > 0:
    valor[pq_] = - prod1[pq_] \
                 - prod2[pq_] \
                 - prod3[pq_] \
                 - (1 - s0) * Yshunts[pq_] * Up[0, pq_] * abs(Us0[pq]) ** 2 \
                 + (1 - s0) * (Pfi[pq] - Qfi[pq] * 1j) * Xp[0, pq_]
if npv > 0:
    valor[pv_] = - prod1[pv_] \
                 - prod2[pv_] \
                 - prod3[pv_] \
                 - (1 - s0) * Yshunts[pv_] * Up[0, pv_] * abs(Us0[pv]) ** 2 \
                 + (1 - s0) * Pfi[pv] * Xp[0, pv_]

    RHS = np.r_[valor.real, valor.imag, W[pv_] / abs(Us0[pv]) ** 2 - 1]
else:
    RHS = np.r_[valor.real, valor.imag]

Gf = np.real(Ybtildered)  # part real de la matriu simètrica reduïda
Bf = np.imag(Ybtildered)  # part imaginària de la matriu simètrica reduïda

gamma_re = diags(2 * np.real(gamma[:]))  # matriu diagonal amb part real
gamma_im = diags(2 * np.imag(gamma[:]))  # matriu diagonal amb part imaginària

VRE = coo_matrix((2 * Up_re[0, pv_], (np.arange(npv), pv_)), shape=(npv, npqpv)).tocsc()
VIM = coo_matrix((2 * Up_im[0, pv_], (np.arange(npv), pv_)), shape=(npv, npqpv)).tocsc()
XIM = coo_matrix((-Xp_im[0, pv_], (pv_, np.arange(npv))), shape=(npqpv, npv)).tocsc()
XRE = coo_matrix((Xp_re[0, pv_], (pv_, np.arange(npv))), shape=(npqpv, npv)).tocsc()
EMPTY = csc_matrix((npv, npv))

M1 = np.copy(Gf)
M2 = np.copy(-Bf)
M3 = np.copy(Bf)
M4 = np.copy(Gf)

for i in range(npqpv):
    for j in range(npqpv):
        if i == j:
            M1[i, j] += np.real(2 * gamma[i])  # emplenar amb gamma
            M3[i, j] += np.imag(2 * gamma[i])

MAT = vstack((hstack((M1, M2, XIM)),
              hstack((M3, M4, XRE)),
              hstack((VRE, VIM, EMPTY))), format='csc')

MAT_LU = factorized(MAT.tocsc())  # factoritzar, només cal una vegada
LHS = MAT_LU(RHS)

Up_re[1, :] = LHS[:npqpv]
Up_im[1, :] = LHS[npqpv: 2 * npqpv]
Qp[1, pv_] = LHS[2 * npqpv:]

Up[1, :] = Up_re[1, :] + Up_im[1, :] * 1j
Xp[1, :] = - np.conj(Up[1, :]) * Xp[0, :] / np.conj(Up[0, :])
Xp_re[1, :] = np.real(Xp[1, :])
Xp_im[1, :] = np.imag(Xp[1, :])
# .......................FI TERMES [1] ........................

# .......................CONVOLUCIONS ........................


def convxv(xp, up, i, cc):
    suma = 0
    for k in range(1, cc):
        suma = suma + xp[k, i] * np.conj(up[cc-k, i])
    return suma


def convqx(qp, xp, i, cc):
    suma = 0
    for k in range(1, cc):
        suma = suma + qp[k, i] * xp[cc-k, i]
    return suma


def convu(up, i, cc):
    suma = 0
    for k in range(1, cc):
        suma = suma + up[k, i] * np.conj(up[cc-k, i])
    return suma
# .......................FI CONVOLUCIONS ........................

# .......................TERMES [2] ........................
prod2 = np.dot(Yahatred[pqpv_, :], Up[1, :])
prod3 = np.dot(Yahatw[pqpv_, :], Upw[1, :])

if npq > 0:
    valor[pq_] = - prod2[pq_] \
                 - prod3[pq_] \
                 - (1 - s0) * Yshunts[pq_] * Up[1, pq_] * abs(Us0[pq]) ** 2 \
                 + (1 - s0) * (Pfi[pq] - Qfi[pq] * 1j) * Xp[1, pq_] \
                 + s0 * (Pfi[pq] - Qfi[pq] * 1j) * (- convxv(Xp, Up, pq_, 2))
if npv > 0:
    valor[pv_] = - prod2[pv_] \
                 - prod3[pv_] \
                 - (1 - s0) * Yshunts[pv_] * Up[1, pv_] * abs(Us0[pv]) ** 2 \
                 + (1 - s0) * Pfi[pv] * Xp[1, pv_] \
                 - convqx(Qp, Xp, pv_, 2) * 1j \
                 + gamma[pv_] * (- convxv(Xp, Up, pv_, 2))
    RHS = np.r_[valor.real, valor.imag, np.real(-convu(Up, pv_, 2))]
else:
    RHS = np.r_[valor.real, valor.imag]

LHS = MAT_LU(RHS)

Up_re[2, :] = LHS[:npqpv]
Up_im[2, :] = LHS[npqpv: 2 * npqpv]
Qp[2, pv_] = LHS[2 * npqpv:]

Up[2, :] = Up_re[2, :] + Up_im[2, :] * 1j
Xp[2, :] = - convx(Up, Xp, range_pqpv, 2) / np.conj(Up[0, :])
Xp_re[2, :] = np.real(Xp[2, :])
Xp_im[2, :] = np.imag(Xp[2, :])
# .......................FI TERMES [2] ........................

# .......................TERMES [c>=3] ........................
for c in range(3, prof_pw):
    prod2 = np.dot(Yahatred[pqpv_, :], Up[c-1, :])

    if npq > 0:
        valor[pq_] = - prod2[pq_] \
                     - (1 - s0) * Yshunts[pq_] * Up[c-1, pq_] * abs(Us0[pq]) ** 2 \
                     + (1 - s0) * (Pfi[pq] - Qfi[pq] * 1j) * Xp[c-1, pq_] \
                     + s0 * (Pfi[pq] - Qfi[pq] * 1j) * (- convxv(Xp, Up, pq_, c))
    if npv > 0:
        valor[pv_] = - prod2[pv_] \
                     - (1 - s0) * Yshunts[pv_] * Up[c-1, pv_] * abs(Us0[pv]) ** 2 \
                     + (1 - s0) * Pfi[pv] * Xp[c-1, pv_] \
                     - convqx(Qp, Xp, pv_, c) * 1j \
                     + gamma[pv_] * (- convxv(Xp, Up, pv_, c))
        RHS = np.r_[valor.real, valor.imag, np.real(-convu(Up, pv_, c))]
    else:
        RHS = np.r_[valor.real, valor.imag]

    LHS = MAT_LU(RHS)

    Up_re[c, :] = LHS[:npqpv]
    Up_im[c, :] = LHS[npqpv: 2 * npqpv]
    Qp[c, pv_] = LHS[2 * npqpv:]

    Up[c, :] = Up_re[c, :] + Up_im[c, :] * 1j
    Xp[c, :] = - convx(Up, Xp, range_pqpv, c) / np.conj(Up[0, :])
    Xp_re[c, :] = np.real(Xp[c, :])
    Xp_im[c, :] = np.imag(Xp[c, :])

Upfi = np.sum(Up, axis=0)  # tensió prima amb el sumatori

Upfipa = np.zeros(n, dtype=complex)  # tensió prima amb Padé
Qfipv = np.zeros(npqpv, dtype=complex)  # potència reactiva amb Padé
Upfipa[pqpv] = pade4all(prof_pw - 1, Up, 1)
Upfipa[sl] = np.sum(Upw, axis=0)
Up_eps = np.zeros(n, dtype=complex)
Up_ait = np.zeros(n, dtype=complex)
Up_rho = np.zeros(n, dtype=complex)
Up_theta = np.zeros(n, dtype=complex)
Up_eta = np.zeros(n, dtype=complex)
Up_th = np.zeros(n, dtype=complex)
Qp_eps = np.zeros(n, dtype=complex)
Qp_ait = np.zeros(n, dtype=complex)
Qp_rho = np.zeros(n, dtype=complex)
Qp_theta = np.zeros(n, dtype=complex)
Qp_eta = np.zeros(n, dtype=complex)


Ubona = Upfipa * Us0  # tensió final

if npv > 0:
    Qfipv[pv_] = pade4all(prof_pw - 1, Qp[:, pv_], 1)
    Qfi[pv] = Qs0[pv] + Qfipv[pv_]

# ERRORS
S_out = np.asarray(Ubona) * np.conj(np.asarray(np.dot(Ybus, Ubona)))  # computat amb tensions de Padé
S_in = (Pfi[:] + 1j * Qfi[:])
errorx = S_in - S_out  # error de potències
# FI ERRORS
err = max(abs(np.r_[errorx[0, pqpv]]))  # màxim error de potències amb P-W
print('Error P-W amb Padé: ', abs(err))

dfpw = pd.DataFrame(np.c_[np.abs(Ubona), np.angle(Ubona), np.abs(errorx[0, :]), np.real(Sig_re), np.real(Sig_im)],
                        columns=['|V| Padé', 'A. Padé', 'S error', 'Sigma re', 'Sigma im'])
#print(dfpw)

# ALTRES:
# .......................VISUALITZACIÓ DE LA MATRIU ........................
from pylab import *
#MATx és la del sistema del MIH, MAT la del P-W
Amm = abs(MATx.todense())  # passar a densa
figure(1)
f = plt.figure()
imshow(Amm, interpolation='nearest', cmap=plt.get_cmap('gist_heat'))
plt.gray()  # en escala de grisos
plt.show()
plt.spy(Amm)  # en blanc i negre
plt.show()

f.savefig("foo.pdf", bbox_inches='tight')

Bmm = coo_matrix(MATx)  # passar a dispersa
density = Bmm.getnnz() / np.prod(Bmm.shape) * 100  # convertir a percentual
#print('Densitat: ' + str(density) + ' %')


# .......................DOMB-SYKES ........................

bb = np. zeros((prof, npqpv), dtype=complex)
for j in range(npqpv):
    for i in range(3, len(U) - 1):
        #bb[i, j] = np. abs(np.sqrt((U[i+1, j] * U[i-1, j] - U[i, j] ** 2) / (U[i, j] * U[i-2, j] - U[i-1, j] ** 2)))
        bb[i, j] = (U[i, j]) / (U[i-1, j])

vec_1n = np. zeros(prof)
for i in range(3, prof):
    #vec_1n[i] = 1 / i
    vec_1n[i] = i

plt.plot(vec_1n[3:len(U)-1], abs(bb[3:len(U)-1, 28]), 'ro ', markersize=2)
plt.show()

# print(bb[3:len(U) - 2, 28])
# n_ord = abs(bb[len(U) - 2, 28]) - vec_1n[len(U) - 2] * (abs(bb[len(U) - 2, 28]) - abs(bb[len(U) - 3, 28])) / (vec_1n[len(U) - 2] - vec_1n[len(U) - 3])
# print('radi: ' + str(1 / n_ord))

# .......................GRÀFIC SIGMA ........................
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