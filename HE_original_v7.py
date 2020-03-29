
# AUTHORS: Santiago Peñate Vera and Josep Fanals Batllori
# CONTACT:  santiago.penate.vera@gmail.com, u1946589@campus.udg.edu
# thanks to Llorenç Fanals Batllori for his help at coding
# IGUAL QUE HE_original_v5 PERÒ AMB CANVIS A LA YTILDE, SEGUEIXO EL LLIBRE.
# provo el Padé-Weierstrass per slack+PQ, faig debugg

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
pd.set_option("display.precision", 5)
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
Yseries_slack[:,:] = Yseries[:,:]  # també conté les admitàncies amb l'slack

Ytap = np.zeros((n, n), dtype=complex)  # diferència entre Ytapreal i Yseries (aquesta última conté Ys simètrica)

for i in range(nl):
    if df_top.iloc[i, 5] != 1:
        Ytap[df_top.iloc[i, 0], df_top.iloc[i, 0]] += 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) / (df_top.iloc[i, 5] * np.conj(df_top.iloc[i, 5])) - 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)
        Ytap[df_top.iloc[i, 1], df_top.iloc[i, 1]] += 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) - 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)
        Ytap[df_top.iloc[i, 0], df_top.iloc[i, 1]] += - 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) / (np.conj(df_top.iloc[i, 5])) + 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)
        Ytap[df_top.iloc[i, 1], df_top.iloc[i, 0]] += - 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) / (df_top.iloc[i, 5]) + 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)

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

Yshunts_slack = np.zeros(n, dtype=complex)  #inclòs l'slack
Yshunts_slack[:] = Yshunts[:]  # incloent l'slack

df = pd.DataFrame(data=np.c_[Yshunts.imag, vec_Pi, vec_Qi, vec_Vi],
                  columns=['Ysh', 'P0', 'Q0', 'V0'])

print(df)

Yslack = np.zeros((n, n), dtype=complex)
Yslack = Yseries_slack[:, sl]

# --------------------------- INITIAL DATA: BUSES INFORMATION. DONE

# --------------------------- PREPARING IMPLEMENTATION
prof = 6  # depth

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
Ytaps = Ytap[np.ix_(pqpv, pqpv)]  # Ytap reduïda als busos pq i pv
Ytapslack = Ytap[np.ix_(pqpv, sl)]  # agafo la columna per llavors fer-la servir al RHS[1] i RHS[2]
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
range_pqpv = np.arange(npqpv)
valor = np.zeros(npqpv, dtype=complex)

prod = np.dot((Yslack[pqpv_, :]), V_sl[:])
prod2 = np.dot((Ytaps[pqpv_, :]), U[0, :])

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

RHS = np.r_[valor.real, valor.imag, W[pv_] - 1]

VRE = coo_matrix((2 * U_re[0, pv_], (np.arange(npv), pv_)), shape=(npv, npqpv)).tocsc()  # cada element ha de ser 2
VIM = coo_matrix((2 * U_im[0, pv_], (np.arange(npv), pv_)), shape=(npv, npqpv)).tocsc()  # cada element ha de ser 0
XIM = coo_matrix((-X_im[0, pv_], (pv_, np.arange(npv))), shape=(npqpv, npv)).tocsc()  # cada element ha de ser 0
XRE = coo_matrix((X_re[0, pv_], (pv_, np.arange(npv))), shape=(npqpv, npv)).tocsc()  # cada element ha de ser 1
EMPTY = csc_matrix((npv, npv))

MAT = vstack((hstack((G, -B, XIM)),
              hstack((B, G, XRE)),
              hstack((VRE, VIM, EMPTY))), format='csc')

#print(MAT)
MAT_LU = factorized(MAT.tocsc())
LHS = MAT_LU(RHS)

U_re[1, :] = LHS[:npqpv]
U_im[1, :] = LHS[npqpv:2 * npqpv]
Q[1, pv_] = LHS[2 * npqpv:]

U[1, :] = U_re[1, :] + U_im[1, :] * 1j
X[1, :] = (-X[0, :] * np.conj(U[1, :])) / np.conj(U[0, :])
X_re[1, :] = X[1, :].real
X_im[1, :] = X[1, :].imag


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

# .......................CALCULATION OF TERMS [2] ........................
prod2 = np.dot((Ytaps[pqpv_, :]), U[1, :])
prod3 = np.dot((Ytapslack[pqpv_, :]), V_sl[:])

c = 2

valor[pq_] = - Yshunts[pq_] * U[c-1, pq_] \
             + (vec_P[pq_] - vec_Q[pq_] * 1j) * X[c-1, pq_] \
             - prod2[pq_] \
             - np.sum(Ytapslack[pq_, :], axis=1) * (-1) \
             - prod3[pq_]

valor[pv_] = - Yshunts[pv_] * U[c-1, pv_] \
             + vec_P[pv_] * X[c-1, pv_] \
             - 1j * convQX(Q, X, pv_, c) \
             - prod2[pv_] \
             - np.sum(Ytapslack[pv_, :], axis=1) * (-1) \
             - prod3[pv_]

RHS = np.r_[valor.real, valor.imag, -convV(U, pv_, c)]

LHS = MAT_LU(RHS)

U_re[c, :] = LHS[:npqpv]
U_im[c, :] = LHS[npqpv:2 * npqpv]
Q[c, pv_] = LHS[2 * npqpv:]

U[c, :] = U_re[c, :] + U_im[c, :] * 1j
X[c, :] = - convX(U, X, range_pqpv, c) / np.conj(U[0, :])
X_re[c, :] = X[c, :].real
X_im[c, :] = X[c, :].imag

# .......................CALCULATION OF TERMS [2]. DONE ........................

# .......................CALCULATION OF TERMS [c>=3] ........................
for c in range(3, prof):
    prod2 = np.dot((Ytaps[pqpv_, :]), U[c-1, :])

    valor[pq_] = - Yshunts[pq_] * U[c-1, pq_] \
                 + (vec_P[pq_] - vec_Q[pq_] * 1j) * X[c-1, pq_] \
                 - prod2[pq_]

    valor[pv_] = - Yshunts[pv_] * U[c-1, pv_] \
                 + vec_P[pv_] * X[c-1, pv_] \
                 - 1j * convQX(Q, X, pv_, c) \
                 - prod2[pv_]

    RHS = np.r_[valor.real, valor.imag, -convV(U, pv_, c)]

    LHS = MAT_LU(RHS)

    U_re[c, :] = LHS[:npqpv]
    U_im[c, :] = LHS[npqpv:2 * npqpv]
    Q[c, pv_] = LHS[2 * npqpv:]

    U[c, :] = U_re[c, :] + U_im[c, :] * 1j
    X[c, :] = - convX(U, X, range_pqpv, c) / np.conj(U[0, :])
    X_re[c, :] = X[c, :].real
    X_im[c, :] = X[c, :].imag

# .......................CALCULATION OF TERMS [c>3]. DONE ........................

# .......................RESULTATS ........................
Pfi = np.zeros(n, dtype=complex)
Qfi = np.zeros(n, dtype=complex)
U_sum = np.zeros(n, dtype=complex)
U_pa = np.zeros(n, dtype=complex)
U_sig = np.zeros(n, dtype=complex)
U_th = np.zeros(n, dtype=complex)
U_ait = np.zeros(n, dtype=complex)
U_eps = np.zeros(n, dtype=complex)
U_rho = np.zeros(n, dtype=complex)
U_theta = np.zeros(n, dtype=complex)
U_eta = np.zeros(n, dtype=complex)
Sig_re = np.zeros(n, dtype=complex)
Sig_im = np.zeros(n, dtype=complex)

Ybus = Yseries_slack + diags(Yshunts_slack) + Ytap

from Funcions import pade4all, epsilon, eta, theta, aitken, Sigma_funcO, rho, thevenin_funcX2

#SUMA
U_sum[pqpv] = np.sum(U[:, pqpv_], axis=0)
U_sum[sl] = V_sl
#FI SUMA

#PADÉ
Upa = pade4all(prof-1, U[:, :], 1)
Qpa = pade4all(prof-1, Q[:, pv_], 1)
U_pa[sl] = V_sl
U_pa[pqpv] = Upa
Pfi[pqpv] = vec_P[pqpv_]
if npq > 0:
    Qfi[pq] = vec_Q[pq_]
if npv > 0:
    Qfi[pv] = Qpa
Pfi[sl] = np.nan
Qfi[sl] = np.nan
#FI PADÉ

limit = 10  # límit per tal que recurrents no tirin error. Si pocs busos, millor límit baix
if limit > prof:
    limit = prof - 1
Ux = np.copy(U)

#SIGMA
Sig_re[pqpv] = np.real(Sigma_funcO(Ux, X, prof-1, V_sl))
Sig_im[pqpv] = np.imag(Sigma_funcO(Ux, X, prof-1, V_sl))
Sig_re[sl] = np.nan
Sig_im[sl] = np.nan
s_p = 1 / (2 * (abs(np.real(Sig_re) + np.real(Sig_im) * 1j) - np.real(Sig_re)))
s_n = - 1 / (2 * (abs(np.real(Sig_re) + np.real(Sig_im) * 1j) + np.real(Sig_re)))
#FI SIGMA

#THÉVENIN
for i in range(npqpv):  # provar si pels PV també va
    U_th[i] = thevenin_funcX2(Ux[:limit, i], X[:limit, i], 1)
U_th[pqpv] = U_th[pqpv_]
U_th[sl] = np.nan
#FI THÉVENIN

#AITKEN
for i in range(npqpv):
    U_ait[i] = aitken(Ux[:, i], limit)
U_ait[pqpv] = U_ait[pqpv_]
U_ait[sl] = np.nan
#FI AITKEN

#EPSILON ACCELERADES
for i in range(npqpv):
    U_eps[i] = epsilon(sum(Ux[:, i]), limit, Ux[:, i])
U_eps[pqpv] = U_eps[pqpv_]  # treballo amb Ux perquè U sembla canviar a algun moment
U_eps[sl] = np.nan
#FI EPSILON ACCELERADES

#RHO
for i in range(npqpv):
    U_rho[i] = rho(Ux[:, i], limit)
U_rho[pqpv] = U_rho[pqpv_]
U_rho[sl] = np.nan
#FI RHO

#THETA
for i in range(npqpv):
    U_theta[i] = theta(Ux[:, i], limit)
U_theta[pqpv] = U_theta[pqpv_]
U_theta[sl] = np.nan
#FI THETA

#ETA
for i in range(npqpv):
    U_eta[i] = eta(Ux[:, i], limit)
U_eta[pqpv] = U_eta[pqpv_]
U_eta[sl] = np.nan
#FI ETA

#ERRORS
S_out = np.asarray(U_pa) * np.conj(np.asarray(np.dot(Ybus, U_pa)))  # computat amb tensions de Padé
S_in = (Pfi[:] + 1j * Qfi[:])
error = S_in - S_out  # mismatch de potències
#FI ERRORS

df = pd.DataFrame(np.c_[np.abs(U_sum), np.angle(U_sum), np.abs(U_pa), np.angle(U_pa), np.abs(U_th),
                        np.abs(U_eps), np.angle(U_eps), np.abs(U_ait), np.angle(U_ait), np.abs(U_rho),
                        np.angle(U_rho), np.abs(U_theta), np.angle(U_theta), np.abs(U_eta), np.angle(U_eta),
                        np.real(Pfi), np.real(Qfi), np.abs(error[0, :]), np.real(Sig_re), np.real(Sig_im), s_p, s_n],
                        columns=['|V| sum', 'A. sum', '|V| Padé', 'A. Padé', '|V| Thévenin', '|V| Epsilon',
                                 'A. Epsilon', '|V| Aitken', 'A. Aitken', '|V| Rho', 'A. Rho', '|V| Theta', 'A. Theta',
                                 '|V| Eta', 'A. Eta', 'P', 'Q', 'S error', 'Sigma re', 'Sigma im', 's+', 's-'])

print(df)

err = max(abs(np.r_[error[0, pqpv]]))  # màxim error de potències
print('Error màxim: ' + str(err))







#### PADÉ-WEIERSTRASS ####

def vector_s0(vec, s_0):  # per calcular V(s_0)
    suma = 0
    for k in range(len(vec)):
        suma += vec[k] * s_0 ** k
    return suma

s0 = 0.53
prof_pw = 30

Up = np.zeros((prof_pw, npqpv), dtype=complex)  # matriu on es guarden els coeficients de les tensions no slack primes
Up_re = np.zeros((prof_pw, npqpv), dtype=float)
Up_im = np.zeros((prof_pw, npqpv), dtype=float)
Xp = np.zeros((prof_pw, npqpv), dtype=complex)
Xp_re = np.zeros((prof_pw, npqpv), dtype=float)
Xp_im = np.zeros((prof_pw, npqpv), dtype=float)
Qp = np.zeros((prof_pw, npqpv), dtype=complex)
Upw = np.zeros((prof_pw, nsl), dtype=complex)  # matriu on es guarden els coeficients de les tensions slack primes
Us0 = np.zeros(n, dtype=complex)  # vector amb totes les V(s0)
Qs0 = np.zeros(n, dtype=complex)  # vector amb totes les Q(s0) dels busos PV, és clar

V_sl = np.asarray(V_sl)

Us0[sl] = 1 + s0 * V_sl[sl] - s0 * 1
Upw[0] = 1
Upw[1, sl] = (1-s0)*(V_sl[:]-1)/(1+s0*(V_sl[:]-1))  # V prima de l'slack amb tots els coef, quasi tots són 0

Us0[pqpv] = vector_s0((U[:, pqpv_]), s0)

if npv > 0:
    Qs0[pv] = vector_s0((Q[:, pv_]), s0)

Yahat = np.zeros((n, n), dtype=complex)  # és l'asimètrica, la Y^(a) amb el circumflex
Ybhat = np.zeros((n, n), dtype=complex)  # és la simètrica, la Y^(b) amb el circumflex

Yahat[:, :] = Ytap[:, :]
Ybhat[:, :] = Yseries_slack[:, :]

for i in range(n):
    if i not in sl:  # per la fila de l'slack no cal fer-ho
        for j in range(n):
            Yahat[i, j] = Yahat[i, j] * Us0[j] * np.conj(Us0[i])
            Ybhat[i, j] = Ybhat[i, j] * Us0[j] * np.conj(Us0[i])

gamma = np.zeros(npqpv, dtype=complex)

if npq > 0:
    gamma[pq_] = s0 * (Pfi[pq] - Qfi[pq] * 1j)
if npv > 0:
    gamma[pv_] = s0 * Pfi[pv] - Qs0[pv] * 1j

Ybtilde = np.zeros((n, n), dtype=complex)  # és la simètrica, la Y^(b) amb la tilde
Ybtilde[:, :] = Ybhat[:, :]

if npq > 0:
    Ybtilde[pq, pq] += +s0 * Yshunts_slack[pq] * abs(Us0[pq]) ** 2 - gamma[pq_]
if npv > 0:
    Ybtilde[pv, pv] += +s0 * Yshunts_slack[pv] * abs(Us0[pv]) ** 2 - gamma[pv_]

Ybhatsum = np.sum(Ybhat, axis=1)

"""
##NOVA MANERA, COM DIU EL LLIBRE:
for i in range(n):
    for j in range(n):
        if i == j:
            Ybtilde[i, j] += - Ybhatsum[i]
"""


#!!!!!!!!!!!!!!!!!ajustament important: s=s0+s'(1-s0) -> Yb+sYa=(Yb+s0*Ya)+s'(1-s0)Ya!!!!!!!!!!!!!!!!!!!
### o millor incrusto amb s' i canvio una mica l'estructura del problema...

"""
Ybtilde[:, :] += s0 * Yahat[:, :]
Yahat[:, :] = (1 - s0) * Yahat[:, :]
"""

"""
for i in range(n):  # per agafar el que he modificicat a Yb!! ho trec d'una banda i li poso a l'altra
    for j in range(n):
        if i == j:
            Yahat[i, j] += + Ybhatsum[i]
"""

##això d'aquí sota ho he afegit ara
if npq > 0:  # per agafar el que he modificicat a Yb!! ho trec d'una banda i li poso a l'altra
    Yahat[pq, pq] -= +s0 * Yshunts_slack[pq] * abs(Us0[pq]) ** 2 - gamma[pq_]
if npv > 0:
    Yahat[pv, pv] -= +s0 * Yshunts_slack[pv] * abs(Us0[pv]) ** 2 - gamma[pv_]


print('ybtilde', Ybtilde)
print('yahat', Yahat)

#print('suma', np.sum(Ybtilde, axis=1))  # ara sí que suma 0 (menys l'slack, és clar. ja està bé així)

########################  ORDRE [0]  ##########################
Up[0, :] = 1
Qp[0, :] = 0

Up_re[0, :] = np.real(Up[0, :])
Up_im[0, :] = np.imag(Up[0, :])
Xp[0, :] = 1 / Up[0, :]
Xp_re[0, :] = np.real(Xp[0, :])
Xp_im[0, :] = np.imag(Xp[0, :])

########################  FI ORDRE [0]  ##########################


#Redueixo matrius. Ya sense slack, Ya amb slack, Yb sense slack, Yb amb slack

Yahatred = Yahat[np.ix_(pqpv, pqpv)]  # asimètrica sense slack
Yahatw = Yahat[np.ix_(pqpv, sl)]  # asimètrica de l'slack
Ybtildered = Ybtilde[np.ix_(pqpv, pqpv)]  # simètrica sense slack
Ybtildew = Ybtilde[np.ix_(pqpv, sl)]  # simètrica amb slack

#print(Yahatred)
#print(Yahatw)
#print(Ybtildered)
#print(Ybtildew)

########################  ORDRE [1]  ##########################, elimino prod2 i prod 3 perquè s'anul·la
prod1 = np.dot(Ybtildew[pqpv_, :], Upw[1, :])
#prod2 = np.dot(Yahatred[pqpv_, :], Up[0, :])
#prod3 = np.dot(Yahatw[pqpv_, :], Upw[0, :])

if npq > 0:
    valor[pq_] = - prod1[pq_] \
                 - (1 - s0) * Yshunts[pq_] * Up[0, pq_] * abs(Us0[pq]) ** 2 \
                 + (1 - s0) * (Pfi[pq] - Qfi[pq] * 1j) * Xp[0, pq_]

if npv > 0:
    valor[pv_] = - prod1[pv_] \
                 - (1 - s0) * Yshunts[pv_] * Up[0, pv_] * abs(Us0[pv]) ** 2 \
                 + (1 - s0) * Pfi[pv] * Xp[0, pv_]

    RHS = np.r_[valor.real, valor.imag, W[pv_] / abs(Us0[pv]) ** 2 - 1]
else:
    RHS = np.r_[valor.real, valor.imag]


Gf = np.real(Ybtildered)  # real parts of Yij
Bf = np.imag(Ybtildered)  # imaginary parts of Yij

gamma_re = diags(2 * np.real(gamma[:]))
gamma_im = diags(2 * np.imag(gamma[:]))

#print('gam-re', gamma_re)
#print('gam-im', gamma_im)

VRE = coo_matrix((2 * Up_re[0, pv_], (np.arange(npv), pv_)), shape=(npv, npqpv)).tocsc()  # cada element ha de ser 2
VIM = coo_matrix((2 * Up_im[0, pv_], (np.arange(npv), pv_)), shape=(npv, npqpv)).tocsc()  # cada element ha de ser 0
XIM = coo_matrix((-Xp_im[0, pv_], (pv_, np.arange(npv))), shape=(npqpv, npv)).tocsc()  # cada element ha de ser 0
XRE = coo_matrix((Xp_re[0, pv_], (pv_, np.arange(npv))), shape=(npqpv, npv)).tocsc()  # cada element ha de ser 1
EMPTY = csc_matrix((npv, npv))

M1 = np.zeros((npqpv, npqpv), dtype=float)
M2 = np.zeros((npqpv, npqpv), dtype=float)
M3 = np.zeros((npqpv, npqpv), dtype=float)
M4 = np.zeros((npqpv, npqpv), dtype=float)

M1[:, :] = Gf[:, :]
M2[:, :] = - Bf[:, :]
M3[:, :] = Bf[:, :]
M4[:, :] = Gf[:, :]

for i in range(npqpv):
    for j in range(npqpv):
        if i == j:
            M1[i, j] += np.real(2 * gamma[i])
            M3[i, j] += np.imag(2 * gamma[i])

#print('M1', M1, M2, M3, M4)

MAT = vstack((hstack((M1, M2, XIM)),
              hstack((M3, M4, XRE)),
              hstack((VRE, VIM, EMPTY))), format='csc')

#print('mat',MAT)

MAT_LU = factorized(MAT.tocsc())
LHS = MAT_LU(RHS)

Up_re[1, :] = LHS[:npqpv]
Up_im[1, :] = LHS[npqpv: 2 * npqpv]
Qp[1, pv_] = LHS[2 * npqpv:]

Up[1, :] = Up_re[1, :] + Up_im[1, :] * 1j
Xp[1, :] = - np.conj(Up[1, :]) * Xp[0, :] / np.conj(Up[0, :])
Xp_re[1, :] = np.real(Xp[1, :])
Xp_im[1, :] = np.imag(Xp[1, :])


########################  FI ORDRE [1]  ##########################

def convXV(Xp, Up, i, c):
    suma = 0
    for k in range(1, c):
        suma = suma + Xp[k, i] * np.conj(Up[c-k, i])
    return suma

def convQX(Qp, Xp, i, c):
    suma = 0
    for k in range(1, c):
        suma = suma + Qp[k, i] * Xp[c-k, i]
    return suma

def convU(Up, i, c):
    suma = 0
    for k in range(1, c):
        suma = suma + Up[k, i] * np.conj(Up[c-k, i])
    return suma

########################  ORDRE [2]  ##########################

prod1 = np.dot(Ybtildew[pqpv_, :], Upw[2, :])  # seran tot 0
#prod2 = np.dot(Yahatred[pqpv_, :], Up[1, :])
#prod3 = np.dot(Yahatw[pqpv_, :], Upw[1, :])


if npq > 0:
    valor[pq_] = - prod1[pq_] \
                 - (1 - s0) * Yshunts[pq_] * Up[1, pq_] * abs(Us0[pq]) ** 2 \
                 + (1 - s0) * (Pfi[pq] - Qfi[pq] * 1j) * Xp[1, pq_] \
                 + s0 * (Pfi[pq] - Qfi[pq] * 1j) * (- convXV(Xp, Up, pq_, 2))

if npv > 0:
    valor[pv_] = - prod1[pv_] \
                 - (1 - s0) * Yshunts[pv_] * Up[1, pv_] * abs(Us0[pv]) ** 2 \
                 + (1 - s0) * Pfi[pv] * Xp[1, pv_] \
                 - convQX(Qp, Xp, pv_, 2) * 1j \
                 + gamma[pv_] * (- convXV(Xp, Up, pv_, 2))

    RHS = np.r_[valor.real, valor.imag, np.real(-convU(Up, pv_, 2))]

else:
    RHS = np.r_[valor.real, valor.imag]

LHS = MAT_LU(RHS)

Up_re[2, :] = LHS[:npqpv]
Up_im[2, :] = LHS[npqpv: 2 * npqpv]
Qp[2, pv_] = LHS[2 * npqpv:]

Up[2, :] = Up_re[2, :] + Up_im[2, :] * 1j
Xp[2, :] = - convX(Up, Xp, range_pqpv, 2) / np.conj(Up[0, :])
Xp_re[2, :] = np.real(Xp[2, :])
Xp_im[2, :] = np.imag(Xp[2, :])


########################  FI ORDRE [2]  ##########################

########################  ORDRE [c]  ##########################
for c in range(3, prof_pw):
    #prod1 = np.dot(Ybtildew[pqpv_, :], Upw[c, :])  # seran tot 0
    #prod2 = np.dot(Yahatred[pqpv_, :], Up[c-1, :])
    #prod3 = np.dot(Yahatw[pqpv_, :], Upw[c-1, :])  # seran tot 0

    if npq > 0:
        valor[pq_] = 0 \
                     - (1 - s0) * Yshunts[pq_] * Up[c-1, pq_] * abs(Us0[pq]) ** 2 \
                     + (1 - s0) * (Pfi[pq] - Qfi[pq] * 1j) * Xp[c-1, pq_] \
                     + s0 * (Pfi[pq] - Qfi[pq] * 1j) * (- convXV(Xp, Up, pq_, c))

    if npv > 0:
        valor[pv_] = - prod1[pv_] \
                     - prod2[pv_] \
                     - prod3[pv_] \
                     - (1 - s0) * Yshunts[pv_] * Up[c-1, pv_] * abs(Us0[pv]) ** 2 \
                     + (1 - s0) * Pfi[pv] * Xp[c-1, pv_] \
                     - convQX(Qp, Xp, pv_, c) * 1j \
                     + gamma[pv_] * (- convXV(Xp, Up, pv_, c))

        RHS = np.r_[valor.real, valor.imag, np.real(-convU(Up, pv_, c))]
    else:
        RHS = np.r_[valor.real, valor.imag]

    LHS = MAT_LU(RHS)

    Up_re[c, :] = LHS[:npqpv]
    Up_im[c, :] = LHS[npqpv: 2 * npqpv]
    Qp[c, pv_] = LHS[2 * npqpv:]

    Up[c, :] = Up_re[c, :] + Up_im[c, :] * 1j
    Xp[c, :] = - convX(Up, Xp, range_pqpv, c) / np.conj(Up[0, :])
    Xp_re[c, :] = np.real(Xp[c, :])
    Xp_im[c, :] = np.imag(Xp[c, :])



Upfi = np.sum(Up, axis=0)

Upfipa = np.zeros(n, dtype=complex)
Upfipa[pqpv] = pade4all(prof_pw - 1, Up, 1)
Upfipa[sl] = np.sum(Upw, axis=0)

#print('Up',Upfi)
#print('Us0',Us0)

#print('U final ', abs(Upfi * Us0[pqpv]))
#print('Angle ufinal', angle(Upfi * Us0[pqpv]))

## miro si al final compleixo amb l'equació del principi:


A1 = 25 * 1j * np.conj(Us0[1]) * Us0[0] * (Upw[0, 0] + Upw[1, 0]) + -25 * 1j * abs(Us0[1]) ** 2 * Upfi[0]

A2 = 0.95 * (-1 + 0.5 * 1j) * (1 / np.conj(Upfi[0]))

#print(A1)
#print(A2)
#print(A1 - A2)

#print('Up', Up, Up_re, Up_im)
#print('Xp', Xp, Xp_re, Xp_im)
#print('Qp', Qp)
#print('Upw', Upw)
#print('Uso, Qs0', Us0, Qs0)


#print(Ytap)
#print(Yseries_slack)

Ubona = Upfipa * Us0

Qfipv = np.zeros(npqpv, dtype=complex)
for i in range(npqpv):
    if i in pv_:
        Qfipv[i] = np.sum(Qp[:, i], axis=0)

if npv > 0:
    Qfi[pv] = Qs0[pv] + Qfipv[pv_]

print('Ubona: ', Ubona)
print('Qfi: ', Qfi)


#ERRORS
S_out = np.asarray(Ubona) * np.conj(np.asarray(np.dot(Ybus, Ubona)))  # computat amb tensions de Padé
S_in = (Pfi[:] + 1j * Qfi[:])
error = S_in - S_out  # mismatch de potències
#FI ERRORS
err = max(abs(np.r_[error[0, pqpv]]))  # màxim error de potències
print('Error P-W: ', abs(err))

