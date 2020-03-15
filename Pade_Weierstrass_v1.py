def valor_num(U, s):
    val = 0
    for k in range(len(U)):
        val += U[k] * s**k
    return val

def valor_vec(U, s):
    val = []
    for k in range(len(U)):
        val.append(U[k] * s**k)
    return val

def check_conv(vec_val):
    converg = True
    compt = 0  # per tal que no només una vegada no convergeixi
    limit = 1
    i = 0
    while converg == True and i < len(vec_val) - 1:
        if abs(vec_val[i+1]) > abs(vec_val[i]): # faltaven els absoluts!!
            compt += compt
        if compt == limit:
            converg = False
        i += 1
    return converg

def s_conv(U, tol):
    s = 0.9
    s_inf = 0
    s_sup = 10
    while abs(s-s_sup) > tol or abs(s-s_inf) > tol:
        vec_val = valor_vec(U,s)
        conv = check_conv(vec_val)
        if conv == True:
            s_inf = s
            s = (s_inf + s_sup) / 2
        else:
            s_sup = s
            s = (s_inf + s_sup) / 2
    return s

