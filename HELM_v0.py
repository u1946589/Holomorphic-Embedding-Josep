#AUTHOR: Josep Fanals Batllori
#CONTACT: u1946589@campus.udg.edu
# --------------------------- LIBRARIES
import numpy as np
import pandas as pd
# --------------------------- END LIBRARIES
# --------------------------- INITIAL DATA: Y, SHUNTS AND Y0i
df_top = pd.read_excel('Dades_v1.xlsx', sheet_name='Topology')  # dataframe of the topology
num_busos = 0 #number of buses initialized to 0
busos_coneguts = np.zeros(0, dtype=int) #vector to store the indices of the found buses
def trobar(element, vector): #function to check if an element is in a vector
    if element in vector:
        return True
    else:
        return False
    
for i in range(df_top.shape[0]):  #go through all rows
    for j in range(0, 2):  #go through 1st and 2nd column to grab the bus' indices
        if not trobar(df_top.iloc[i, j], busos_coneguts): #if the index is new
            num_busos += 1
            busos_coneguts = np.append(busos_coneguts, df_top.iloc[i, j])
n = num_busos
Yx = np.zeros((n, n), dtype=complex) #matrix with all series admittances, also slack bus
for i in range(df_top.shape[0]):  #go through all rows
    Yx[df_top.iloc[i, 0], df_top.iloc[i, 0]] = Yx[df_top.iloc[i, 0], df_top.iloc[i, 0]] + 1 / (
                df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)  # diagonal element
    Yx[df_top.iloc[i, 1], df_top.iloc[i, 1]] = Yx[df_top.iloc[i, 1], df_top.iloc[i, 1]] + 1 / (
                df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)  # diagonal element
    Yx[df_top.iloc[i, 0], df_top.iloc[i, 1]] = Yx[df_top.iloc[i, 0], df_top.iloc[i, 1]] - 1 / (
                df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)  # off diagonal
    Yx[df_top.iloc[i, 1], df_top.iloc[i, 0]] = Yx[df_top.iloc[i, 1], df_top.iloc[i, 0]] - 1 / (
                df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)  # off diagonal
Y = np.zeros((n - 1, n - 1), dtype=complex) #admittance matrix withouth slack bus
for i in range(n - 1):
    for j in range(n - 1):
        Y[i, j] = Yx[i + 1, j + 1] #just ignoring the first row and column
vecx_shunts = np.zeros((n, 1), dtype=complex) #vector with shunt admittances
for i in range(df_top.shape[0]):  # passar per totes les files
    vecx_shunts[df_top.iloc[i, 0], 0] = vecx_shunts[df_top.iloc[i, 0], 0] + df_top.iloc[i, 4] * (-1) * 1j  #B/2 is in column 4. The sign is changed here
    vecx_shunts[df_top.iloc[i, 1], 0] = vecx_shunts[df_top.iloc[i, 1], 0] + df_top.iloc[i, 4] * (-1) * 1j  #B/2 is in column 4. The sign is changed here
vec_shunts = np.zeros((n - 1, 1), dtype=complex) #same vector, just to adapt
for i in range(n - 1):
    vec_shunts[i, 0] = vecx_shunts[i + 1, 0]
# vec_shunts = --vec_shunts  #no need to change the sign, already done
vec_Y0 = np.zeros((n - 1, 1), dtype=complex) #vector with admittances connecting to the slack
for i in range(df_top.shape[0]):  #go through all rows
    if df_top.iloc[i, 0] == 0:  #if slack in the first column
        vec_Y0[df_top.iloc[i, 1] - 1, 0] = 1 / (
                    df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j) #-1 so bus 1 goes to index 0
    elif df_top.iloc[i, 1] == 0:  #if slack in the second column
        vec_Y0[df_top.iloc[i, 0] - 1, 0] = 1 / (df_top.iloc[i, 2] + df_top.iloc[i, 3] * 1j)
G = np.real(Y)  #real parts of Yij
B = np.imag(Y)  #imaginary parts of Yij
# --------------------------- INITIAL DATA: Y, SHUNTS AND Y0i. DONE
# --------------------------- INITIAL DATA: BUSES INFORMATION
print(num_busos)
df_bus = pd.read_excel('Dades_v1.xlsx', sheet_name='Buses')  #dataframe of the buses
if df_bus.shape[0] != num_busos:
    print('Error: número de busos de ''Topologia'' i de ''Busos'' no és igual') #check if number of buses is coherent
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
for i in range(df_bus.shape[0]):  # store the data of both PQ and PV
    vec_P[df_bus.iloc[i, 0] - 1] = df_bus.iloc[i, 1]  # -1 to start at 0
    if df_bus.iloc[i, 4] == 'PQ':
        vec_Q[df_bus.iloc[i, 0] - 1] = df_bus.iloc[i, 2]  # -1 to start at 0
        num_busos_PQ += 1  # identify as PQ bus
        vec_busos_PQ = np.append(vec_busos_PQ, df_bus.iloc[i, 0])
    elif df_bus.iloc[i, 4] == 'PV':
        vec_V[df_bus.iloc[i, 0] - 1] = df_bus.iloc[i, 3]  # -1 to start at 0
        num_busos_PV += 1  # identify as PV bus
        vec_busos_PV = np.append(vec_busos_PV, df_bus.iloc[i, 0])
for i in range(n - 1):
    vec_W[i] = vec_V[i] * vec_V[i]
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
# .......................CALCULATION OF TERMS [0]
vec_U = np.dot(np.linalg.inv(Y), vec_Y0)  # each element is roughly equal to 1
for i in range(n - 1):
    U[0, i] = 1  # force each element to 1. Tiny difference
for i in range(n - 1):
    X[0, i] = 1 / np.conj(U[0, i]) # could force them to be 1 directly
    U_re[0, i] = np.real(U[0, i])  # could force them to be 1 directly
    U_im[0, i] = np.imag(U[0, i])  # could force them to be 0 directly
    X_re[0, i] = np.real(X[0, i])  # could force them to be 1 directly
    X_im[0, i] = np.imag(X[0, i])  # could force them to be 0 directly
# .......................CALCULATION OF TERMS [0]. DONE
    
# .......................CALCULATION OF TERMS [1]
llarg = 2 * num_busos_PQ + 3 * num_busos_PV  # number of unknowns
RHS = np.zeros((llarg, 1), dtype=float)  # vector of the RHS data. Each element has to be real
k = 0  # index that will go through the rows
for i in range(n - 1):  # filling the vector RHS
    if i + 1 in vec_busos_PQ:
        RHS_PQ_i = (V_slack - 1) * vec_Y0[i, 0] + (vec_P[i, 0] - vec_Q[i, 0] * 1j) * X[0, i] + U[0, i] * vec_shunts[i, 0]
        RHS[k] = np.real(RHS_PQ_i)
        RHS[k + 1] = np.imag(RHS_PQ_i)
        k = k + 2
    elif i + 1 in vec_busos_PV:
        RHS_PV_i = (V_slack - 1) * vec_Y0[i, 0] + (vec_P[i, 0]) * X[0, i] + U[0, i] * vec_shunts[i, 0]
        RHS[k] = np.real(RHS_PV_i)
        RHS[k + 1] = np.imag(RHS_PV_i)
        RHS[k + 2] = vec_W[i, 0] - 1
        k = k + 3

mat = np.zeros((llarg, 2 * (n - 1) + num_busos_PV), dtype=complex)  # constant matrix
k = 0  # index that will go through the rows
l = 0  # index that will go through the columns
for i in range(n - 1):  # fill the matrix
    if i + 1 in vec_busos_PQ:
        l = 0
        for j in range(n - 1):
            if j+1 not in vec_busos_PV:
                mat[k, l] = G[i, j]
                mat[k + 1, l] = B[i, j]
                mat[k, l + 1] = -B[i, j]
                mat[k + 1, l + 1] = G[i, j]
                l = l + 2  # 2 columns done
            if j+1 in vec_busos_PV:
                mat[k, l] = G[i, j]
                mat[k + 1, l] = B[i, j]
                mat[k, l + 1] = -B[i, j]
                mat[k + 1, l + 1] = G[i, j]
                mat[k, l + 2] = 0
                mat[k + 1, l + 2] = 0
                l = l + 3  # 3 columns done
        k = k + 2  # 2 rows done
    elif i + 1 in vec_busos_PV:
        l = 0
        for j in range(n - 1):
            if j+1 not in vec_busos_PV:
                mat[k, l] = G[i, j]
                mat[k + 1, l] = B[i, j]
                mat[k, l + 1] = -B[i, j]
                mat[k + 1, l + 1] = G[i, j]
                mat[k + 2, l] = 0
                mat[k + 2, l + 1] = 0
                l = l + 2  # 2 columns done
            if j+1 in vec_busos_PV:
                if j == i:
                    mat[k, l] = G[i, j]
                    mat[k + 1, l] = B[i, j]
                    mat[k, l + 1] = -B[i, j]
                    mat[k + 1, l + 1] = G[i, j]

                    mat[k + 2, l] = 2 * U_re[0, i]
                    mat[k + 2, l + 1] = 2 * U_im[0, i]

                    mat[k, l + 2] = -X_im[0, i]
                    mat[k + 1, l + 2] = X_re[0, i]

                    mat[k + 2, l + 2] = 0
                    l = l + 3  # 3 columns done
                elif j != i:
                    mat[k, l] = G[i, j]
                    mat[k + 1, l] = B[i, j]
                    mat[k, l + 1] = -B[i, j]
                    mat[k + 1, l + 1] = G[i, j]

                    mat[k + 2, l] = 0
                    mat[k + 2, l + 1] = 0

                    mat[k, l + 2] = 0
                    mat[k + 1, l + 2] = 0

                    mat[k + 2, l + 2] = 0
                    l = l + 3
        k = k + 3  # 3 rows done
dfx = pd.DataFrame(mat)
dfx.to_excel('Resultats3.xlsx', index=False, header=False)  # to check the matrix
LHS = np.dot(np.linalg.inv(mat), RHS)  # although mat only has to be inverted once
k = 0
for i in range(n - 1):  # fill unknowns
    if i + 1 in vec_busos_PQ:
        U_re[1, i] = LHS[k, 0]
        U_im[1, i] = LHS[k + 1, 0]
        k = k + 2
    elif i + 1 in vec_busos_PV:
        U_re[1, i] = LHS[k, 0]
        U_im[1, i] = LHS[k + 1, 0]
        Q[0, i] = LHS[k + 2, 0]
        k = k + 3
for i in range(n - 1):  # complete the matrices U and X
    U[1, i] = U_re[1, i] + U_im[1, i] * 1j
    X[1, i] = (-X[0, i] * np.conj(U[1, i])) / np.conj(U[0, i])
    X_re[1, i] = np.real(X[1, i])
    X_im[1, i] = np.imag(X[1, i])
# .......................CALCULATION OF TERMS [1]. DONE
    
# .......................CALCULATION OF TERMS [>=2]
def convX(U,X,c,i): #convolution between U^* and X
    suma=0
    for k in range(1, c+1):  # c+1 perquè arribi fins a c
        suma=suma+np.conj(U[k, i])*X[c-k, i]
    return suma

def sumaPV1(X,Q,c,i): #convolution between X and Q
    suma=0
    for k in range(1,c):
        suma=suma+X[k,i]*Q[c-1-k,i]
    return suma

def sumaPV3(U,c,i): #convolution between U and U
    suma=0
    for k in range(1,c):
        suma=suma+U[k,i]*np.conj(U[c-k,i])
    return suma

for c in range(2,prof): #c defines the current depth
    RHS = np.zeros((llarg, 1), dtype=complex) #is real but a warning appears if it is not defined as complex
    k = 0
    for i in range(n - 1): #fill the vector RHS
        if i + 1 in vec_busos_PQ:
            RHS[k] = np.real((vec_P[i, 0] - vec_Q[i, 0] * 1j) * X[c-1, i] + U[c-1, i] * vec_shunts[i, 0])
            RHS[k + 1] = np.imag((vec_P[i, 0] - vec_Q[i, 0] * 1j) * X[c-1, i] + U[c-1, i] * vec_shunts[i, 0])
            k = k + 2
        elif i + 1 in vec_busos_PV:
            RHS[k] = np.real(sumaPV1(X,Q,c,i)*(-1)*1j+U[c-1,i]*vec_shunts[i,0]+X[c-1,i]*vec_P[i,0]) #afegit això últim!!
            RHS[k+1]=np.imag(sumaPV1(X,Q,c,i)*(-1)*1j+U[c-1,i]*vec_shunts[i,0]+X[c-1,i]*vec_P[i,0]) #afegit això últim!!
            RHS[k+2]=-sumaPV3(U,c,i)
            k = k + 3
    LHS = np.dot(np.linalg.inv(mat), RHS) #no need to invert another time the matrix mat!
    k = 0
    for i in range(n - 1):  #grab the unknowns
        if i + 1 in vec_busos_PQ:
            U_re[c, i] = LHS[k, 0]
            U_im[c, i] = LHS[k + 1, 0]
            k = k + 2
        elif i + 1 in vec_busos_PV:
            U_re[c, i] = LHS[k, 0]
            U_im[c, i] = LHS[k + 1, 0]
            Q[c-1, i] = LHS[k + 2, 0]
            k = k + 3
    for i in range(n - 1):  #complete the matrices
        U[c, i] = U_re[c, i] + U_im[c, i] * 1j
        X[c,i]=-convX(U,X,c,i)/ np.conj(U[0, i])
        X_re[c, i] = np.real(X[c, i])
        X_im[c, i] = np.imag(X[c, i])
matAdf=pd.DataFrame(U)
matAdf.to_excel('Resultats.xlsx', index=False, header=False) #to check the voltages
# .......................CALCULATION OF TERMS [>=2]. DONE

# --------------------------- CHECK DATA
U_final=np.zeros((n-1,1),dtype=complex) #final voltages
for j in range(n-1):
    suma=0
    for i in range(prof):
        suma=suma+U[i,j]
    U_final[j,0]=suma
    
print('V:\n', U_final)
print('\nVabs:\n', abs(U_final)) #absolute value
I_serie=np.dot(Y,U_final) #current flowing through series elements
I_inj_slack=np.zeros((n-1,1),dtype=complex) #current injected by the slack
for i in range(n-1):
    I_inj_slack[i,0]=vec_Y0[i,0]*V_slack
    
I_shunt=np.zeros((n-1,1),dtype=complex) #current through shunts
for i in range(n-1):
    I_shunt[i,0]=-U_final[i]*vec_shunts[i] #change the sign again
I_generada=I_serie-I_inj_slack+I_shunt #current leaving the bus

I_gen2=np.zeros((n-1,1),dtype=complex) #current entering the bus
for i in range(n-1):
    if i + 1 in vec_busos_PQ:
        I_gen2[i,0]=(vec_P[i,0]-vec_Q[i,0]*1j)/np.conj(U_final[i,0])
    elif i + 1 in vec_busos_PV:
        I_gen2[i,0]=(vec_P[i,0]-sum(Q[:,i]*1j))/np.conj(U_final[i,0])
        
print('\nNodal current balance:\n', I_gen2-I_generada) #balance of current. Should be almost 0
Ydf = pd.DataFrame(Q) #to check the unknown reactive power
Ydf.to_excel('Resultats2.xlsx', index=False, header=False)


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

ok = np.isclose(np.abs(U_final[:, 0]), np.abs(V_nr[1:]), atol=1e-3).all()

print('\nVabs:\n', np.abs(U_final[:, 0])) #absolute value

print('Test passed', ok)
if not ok:
    print('It should be:')
    print(np.abs(V_nr[1:]))