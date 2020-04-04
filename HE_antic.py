#NETEJO EL CODI DE LA FORMULACIÓ ANTIGA. COMENTARI EN CATALÀ

# AUTHORS: Santiago Peñate Vera and Josep Fanals Batllori
# CONTACT:  santiago.penate.vera@gmail.com, u1946589@campus.udg.edu
# thanks to Llorenç Fanals Batllori for his help at coding


# --------------------------- LLIBRERIES
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
# --------------------------- FI LLIBRERIA


# al final separar-ho per poder fer córrer amb Numba
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


# --------------------------- CÀRREGA DE DADES INICIALS
df_top = pd.read_excel('IEEE30.xlsx', sheet_name='Topologia')  # dades de topologia
df_bus = pd.read_excel('IEEE30.xlsx', sheet_name='Busos')  # dades dels busos

n = df_bus.shape[0]  # nombre de busos, inclou l'slack
nl = df_top.shape[0]  # nombre de línies

A = np.zeros((n, nl), dtype=int)  # matriu d'incidència, formada per 1, -1 i 0
L = np.zeros((nl, nl), dtype=complex)  # matriu amb les branques
np.fill_diagonal(L, [1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) for i in range(nl)])
A[df_top.iloc[range(nl), 0], range(nl)] = 1
A[df_top.iloc[range(nl), 1], range(nl)] = -1

Yseries = np.dot(np.dot(A, L), np.transpose(A))  # matriu de les branques sèrie
Yseries_real = np.zeros((n, n), dtype=complex)
Yseries_real[:, :] = Yseries[:, :]  # també conté les admitàncies amb l'slack

for i in range(nl):  # emplenar matriu quan hi ha trafo de relació variable
    tap = df_top.iloc[i, 5]
    if tap != 1:
        Yseries[df_top.iloc[i, 0], df_top.iloc[i, 0]] += -1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) \
            + 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) / (tap ** 2)
        Yseries[df_top.iloc[i, 1], df_top.iloc[i, 1]] += -1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) \
            + 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)
        Yseries[df_top.iloc[i, 0], df_top.iloc[i, 1]] += +1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) \
            + -1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) / np.conj(tap)
        Yseries[df_top.iloc[i, 1], df_top.iloc[i, 0]] += +1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) \
            + -1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) / tap

Yseries = csc_matrix(Yseries)  # a dispersa
Yseries_real = csc_matrix(Yseries_real)

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
        V_sl.append(df_bus.iloc[i, 3]*(np.cos(df_bus.iloc[i, 4])+np.sin(df_bus.iloc[i, 4])*1j))
        sl.append(i)

pq = np.array(pq)  # índexs en forma de vector
pv = np.array(pv)
npq = len(pq)  # nombre de busos PQ
npv = len(pv)
if npv > 0 and npq > 0:
    pqpv = np.sort(np.r_[pq, pv])
elif npq > 0:
    pqpv = np.sort(pq)
elif npv > 0:
    pqpv = np.sort(pv)
pq_x = pq  # guardar els índexs originals
pv_x = pv

npqpv = npq + npv  # nombre de busos incògnita
nsl = n - npqpv  # nombre de busos slack

vec_P = vec_Pi[pqpv]  # agafar la part del vector necessària
vec_Q = vec_Qi[pqpv]
vec_V = vec_Vi[pqpv]

vecx_shunts = np.zeros((n, 1), dtype=complex)  # vector amb admitàncies shunt
for i in range(nl):  # de la pestanya topologia
    vecx_shunts[df_top.iloc[i, 0], 0] = vecx_shunts[df_top.iloc[i, 0], 0] + df_top.iloc[
        i, 4] * -1j  # signe canviat
    vecx_shunts[df_top.iloc[i, 1], 0] = vecx_shunts[df_top.iloc[i, 1], 0] + df_top.iloc[
        i, 4] * -1j  # signe canviat
for i in range(n):  # de la pestanya busos
    if df_bus.iloc[i, 6] != 0:
        vecx_shunts[df_bus.iloc[i, 0], 0] += df_bus.iloc[i, 6] * -1j  # signe canviat

vec_shunts = vecx_shunts[pqpv]

df = pd.DataFrame(data=np.c_[vecx_shunts.imag, vec_Pi, vec_Qi, vec_Vi],
                  columns=['Ysh', 'P0', 'Q0', 'V0'])
print(df)

Yslx = np.zeros((n, n), dtype=complex)  # admitàncies que connecten a l'slack

for i in range(nl):
    if df_top.iloc[i, 0] in sl:  # si està a la primera columna
        Yslx[df_top.iloc[i, 1], df_top.iloc[i, 0]] = 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) + \
                                                     Yslx[df_top.iloc[i, 1], df_top.iloc[i, 0]]
    elif df_top.iloc[i, 1] in sl:  # si està a la segona columna
        Yslx[df_top.iloc[i, 0], df_top.iloc[i, 1]] = 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) + \
                                                     Yslx[df_top.iloc[i, 0], df_top.iloc[i, 1]]
Ysl1 = Yslx[:, sl]
Ysl = Ysl1[pqpv, :]
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

vec_W = vec_V * vec_V  # mòdul de les tensions al quadrat
dimensions = 2 * npq + 3 * npv  # nombre d'incògnites
Yred = Yseries[np.ix_(pqpv, pqpv)]  # reduir per a deixar de banda els slack
G = np.real(Yred)  # part real de la matriu
B = np.imag(Yred)  # part imaginària de la matriu

nsl_counted = np.zeros(n, dtype=int)  # nombre de busos slack trobats abans d'un bus
compt = 0
for i in range(n):  # per a índexs que comencin del 0
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
# --------------------------- FI PREPARACIÓ DE LA IMPLEMENTACIÓ

# .......................TERMES [0].......................
if nsl > 1:
    U[0, :] = spsolve(Yred, Ysl.sum(axis=1))
else:
    U[0, :] = spsolve(Yred, Ysl)

X[0, :] = 1 / np.conj(U[0, :])
U_re[0, :] = U[0, :].real
U_im[0, :] = U[0, :].imag
X_re[0, :] = X[0, :].real
X_im[0, :] = X[0, :].imag

# .......................FI TERMES [0] .......................

# ....................... TERMES [1] .......................
valor = np.zeros(npqpv, dtype=complex)

prod = np.dot((Ysl[pqpv_, :]), V_sl[:])  # intensitat que injecten els slack

if npq > 0:
    valor[pq_] = prod[pq_] \
                 - Ysl[pq_].sum(axis=1) + (vec_P[pq_] - vec_Q[pq_] * 1j) * X[0, pq_] \
                 + U[0, pq_] * vec_shunts[pq_, 0]
if npv > 0:
    valor[pv_] = prod[pv_] \
                 - Ysl[pv_].sum(axis=1) \
                 + (vec_P[pv_]) * X[0, pv_] \
                 + U[0, pv_] * vec_shunts[pv_, 0]

    RHS = np.r_[valor.real,
                valor.imag,
                vec_W[pv_] - np.real(U[0, pv_] * np.conj(U[0, pv_]))]  # amb l'equació del mòdul dels PV

    VRE = coo_matrix((2 * U_re[0, pv_], (np.arange(npv), pv_)), shape=(npv, npqpv)).tocsc() # matriu COO a compr.
    VIM = coo_matrix((2 * U_im[0, pv_], (np.arange(npv), pv_)), shape=(npv, npqpv)).tocsc()
    XIM = coo_matrix((-X_im[0, pv_], (pv_, np.arange(npv))), shape=(npqpv, npv)).tocsc()
    XRE = coo_matrix((X_re[0, pv_], (pv_, np.arange(npv))), shape=(npqpv, npv)).tocsc()
    EMPTY = csc_matrix((npv, npv))  # matriu dispera comprimida

    MAT = vstack((hstack((G,   -B,   XIM)),
                  hstack((B,    G,   XRE)),
                  hstack((VRE,  VIM, EMPTY))), format='csc')

else:
    RHS = np.r_[valor.real,
                valor.imag]
    MAT = vstack((hstack((G, -B)),
                  hstack((B, G))), format='csc')

MAT_LU = factorized(MAT.tocsc())  # matriu factoritzada (només cal fer-ho una vegada)
LHS = MAT_LU(RHS)  # obtenir vector d'incògnites

U_re[1, :] = LHS[:npqpv]  # part real de les tensions
U_im[1, :] = LHS[npqpv:2 * npqpv]  # part imaginària de les tensions
if npv > 0:
    Q[0, pv_] = LHS[2 * npqpv:]  # potència reactiva

U[1, :] = U_re[1, :] + U_im[1, :] * 1j
X[1, :] = (-X[0, :] * np.conj(U[1, :])) / np.conj(U[0, :])
X_re[1, :] = X[1, :].real
X_im[1, :] = X[1, :].imag
# .......................FI TERMES [1] .......................

# .......................TERMES [>=2] ........................
range_pqpv = np.arange(npqpv)  # tots els busos ordenats

for c in range(2, prof):  # c és la profunditat actual
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
# .......................FI TERMES [>=2]

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

Ybus = Yseries - diags(vecx_shunts[:, 0])  # matriu d'admitàncies total

from Funcions import pade4all, epsilon, eta, theta, aitken, Sigma_funcO, rho, thevenin_funcX2  # importar funcions

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

limit = 12  # límit per tal que els mètodes recurrents no treballin amb tots els coeficients
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

# EPSILON ACCELERADES
Ux4 = np.copy(U)
Qx4 = np.copy(Q)
for i in range(npqpv):
    U_eps[i] = epsilon(sum(Ux4[:, i]), limit, Ux4[:, i])
    if i in pq_:
        Q_eps[i + nsl_counted[i]] = vec_Q[i]
    elif i in pv_:
        Q_eps[i + nsl_counted[i]] = epsilon(sum(Qx4[:, i]), limit, Qx4[:, i])
U_eps[pqpv] = U_eps[pqpv_]
U_eps[sl] = V_sl
Q_eps[sl] = np.nan
# FI EPSILON ACCELERADES

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
    if i in pq_:
        Q_eta[i + nsl_counted[i]] = vec_Q[i]
    elif i in pv_:
        Q_eta[i + nsl_counted[i]] = eta(Qx7[:, i], limit)
U_eta[pqpv] = U_eta[pqpv_]
U_eta[sl] = V_sl
Q_eta[sl] = np.nan
# FI ETA

S_out = np.asarray(U_pa) * np.conj(np.asarray(np.dot(Ybus.todense(), U_pa)))  # computat amb tensions de Padé
S_in = (Pfi[:] + 1j * Qfi[:])
error = S_in - S_out  # error final de potències

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

# ALTRES:
# .......................VISUALITZACIÓ DE LA MATRIU ........................
from pylab import *
Amm = abs(MAT.todense())  # passar a densa
figure(1)
f = plt.figure()
imshow(Amm, interpolation='nearest', cmap=plt.get_cmap('gist_heat'))
plt.gray()  # en escala de grisos
plt.show()
plt.spy(Amm)  # en blanc i negre
plt.show()

f.savefig("matriu_imatge.pdf", bbox_inches='tight')

Bmm = coo_matrix(MAT)  # passar a dispersa
density = Bmm.getnnz() / np.prod(Bmm.shape) * 100  # convertir a percentual
print('Densitat: ' + str(density) + ' %')

# .......................DOMB-SYKES ........................
bb = np. zeros((prof, npqpv), dtype=complex)
for j in range(npqpv):
    for i in range(3, len(U) - 1):
        #bb[i, j] = np. abs(np.sqrt((U[i+1, j] * U[i-1, j] - U[i, j] ** 2) / (U[i, j] * U[i-2, j] - U[i-1, j] ** 2)))
        bb[i, j] = (U[i, j]) / (U[i-1, j])  # el Domb-Sykes més bàsic

vec_1n = np. zeros(prof)
for i in range(3, prof):
    #vec_1n[i] = 1 / i
    vec_1n[i] = i

plt.plot(vec_1n[3:len(U)-1], abs(bb[3:len(U)-1, 28]), 'ro ', markersize=2)
plt.show()

# print(bb[3:len(U) - 2, 28])
# n_ord = abs(bb[len(U) - 2, 28]) - vec_1n[len(U) - 2] * (abs(bb[len(U) - 2, 28]) - abs(bb[len(U) - 3, 28])) / (vec_1n[len(U) - 2] - vec_1n[len(U) - 3])
# print('radi: ' + str(1 / n_ord))