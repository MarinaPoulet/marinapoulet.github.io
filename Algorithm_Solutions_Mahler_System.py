

# This is an implementation of the Algorithm 1 in the paper "Computing basis of solutions of any Mahler equation" of Faverjon and Poulet.
# We use the notations introduced in this paper.
# A Mahler system is a pair (A,p) where A is a matrix whose entries are rational functions and p is an integer greater than 1.
# Examples of how to use this algorithm are provided at the end of the file 

#########################################################################################################
from sympy import *
z = symbols('z')

from fractions import *
from functools import reduce

import math


#########################################################################################################
# SECTION : Main algorithms

  
# The following function returns a list containing the ramification order d and the matrices Theta and Pbar of Algorithm 1, 
# where Pbar is a truncation of P with (P, Theta) an admissible pair for the Mahler system (A(z**d),p).
# To recover an admissible pair for the initial system, one must consider Pbar(z**(1/d)) and Theta(z**(1/d)).
# To obtain the expansion of P at higher order, one identify the powers of z in the identity P(z)=A(z).inv()* P(z**p) * Theta. Algo1 returns enough coefficients of P to obtain any coefficients of P by induction, using this equation.

def Algo1(A,p):
    B = A.inv()
    m = A.rows
    d = ramif_order(A,p)
    D = A.subs(z,z**d)
    Dinv = B.subs(z,z**d)
    integ = integers(D,p,Dinv,m)  
    vsp = vspaces(D,p,Dinv,m)
    T = matTheta(D,p,Dinv,m,integ,vsp)
    Pbar = matPbar(D,p,Dinv,m,integ,vsp)
    print("The ramification order is d=",d,".")
    print("An admissible pair associated with A(z**d) is (P,\Theta) with \Theta=", T)
    print("and P a Puiseux matrix whose first coefficients are")
    print(Pbar,".")
    return [d,Pbar,T]

# Example : A=Matrix([[z+2,z**2-1],[2,z**2+z-1]]), p=3, Algo1(A,p) returns [2,Matrix([[z/2,z**2-1],[z,z**2-1]]),Matrix([[1,0],[0,1]])]

# When one already knows the integer d=d(A) attached to the system, the algorithm below is quicker than Algo1
def Algo1_with_d(A,p,d):
    B = A.inv()
    m = A.rows
    D = A.subs(z,z**d)
    Dinv = B.subs(z,z**d)
    integ = integers(D,p,Dinv,m)  
    vsp = vspaces(D,p,Dinv,m)
    T = matTheta(D,p,Dinv,m,integ,vsp)
    Pbar = matPbar(D,p,Dinv,m,integ,vsp)
    print("An admissible pair associated with A(z**d) is (P,\Theta) with \Theta=", T)
    print("and P a Puiseux matrix whose first coefficients are")
    print(Pbar,".")
    return [Pbar,T]

# Example : A=Matrix([[z+2,z**2-1],[2,z**2+z-1]]), p=3, Algo1_with_d(A,p,2) returns [Matrix([[z/2,z**2-1],[z,z**2-1]]),Matrix([[1,0],[0,1]])]
# However, Algo1_with_d(A,p,1) does not terminate, for there is no admissible pair with P a matrix of Laurent series.

    
# Let L=[a_0,a_1,...,a_m] be the list of coefficients of a p-Mahler equation. AlgoEq returns a list [A,d,Pbar,Theta] where A is the matrix of the companion system of this equation, and (d,Pbar,Theta) is the output of Algo1(A,p)..
# When one is given an equation, AlgoEq runs faster than Algo1 applied to the companion system for one does not need to use the Cyclic Vector Lemma (function main_cylic_vector_algo).

def AlgoEq(L,p):
    m=len(L)-1
    L2=[]
    for i in range(m):
        L2.append(-L[i]/L[m])
    A = zeros(m,m)
    for i in range(m-1):
        A[i,i+1] = 1 
    for i in range(m):
        A[m-1,i] = L2[i]
    d = ramif_order_eq(L2,p)
    print("The matrix of the associated companion system is ",A)
    ret = [A,d]
    Alg=Algo1_with_d(A,p,d)
    ret.append(Alg[0])
    ret.append(Alg[1])
    return ret

# Example : L=[z,2+3*z**2,1+5*z+z**4], p=5, AlgoEq(L,p) returns [Matrix([[0,1],[-z/(z**4+5*z+1),(-3*z**2-2)/(z**4+5*z+1)]]),4,Matrix([[-z/2,1],[0,-2]]),Matrix([[-z/2,1],[0,-2]])]

# More examples are given at the end of this file.

#########################################################################################################
#########################################################################################################
# SECTION : Useful functions (valuations and series expansions) from the algorithm implemented of [FP22] "An algorithm to recognize regular singular Mahler systems" of Faverjon and Poulet (in Math. Comp., 91 :2905-2928, 2022).

# Valzero(f) computes the valuation at 0 of a rational function f.

def valzero(f): 
    irred = cancel(f) 
    num = numer(irred) 
    den = denom(irred) 
    if f == 0:
        return float("inf") 
    if num.subs(z,0) == 0: 
        a = cancel(num.subs(z,1/z))
        val_num = degree(num, gen=z) - degree(numer(a), gen=z) 
        return int(val_num)
    if den.subs(z,0) == 0:
        b = cancel(den.subs(z,1/z))
        val_den = degree(den, gen=z) - degree(numer(b), gen=z)
        return int(- val_den)
    else:
        return 0
    
# Val(A) computes the valuation at 0 of a matrix A whose entries are rational functions.

def val(A):
    a = min(valzero(A[i,j]) for i in range(A.rows) for j in range(A.cols))      
    return a


# devserie(f,a,b), a<=b integers, returns the list [x_a,...,x_b] where x_i is the coefficient of z^i
# in the Laurent expansion of a rational function f.

def devserie(f,a,b):
    if f == 0:
        return [0]*(b-a+1) 
    n0 = valzero(f)
    if b < n0:
        return [0]*(b-a+1)      
    c = b-n0+1
    g = series(cancel(z**(-n0)*f),z,0,c).removeO()
    h = Poly(g,z).all_coeffs()
    if a < n0:
        h.extend([0]*(n0-a))     
    if n0 < a:
        h = h[0:-(a-n0)]        
    h.reverse()
    while len(h) < b-a+1:
        h.append(0)
    return h
        
# devmat(A,a,b) returns the list of matrices [A_a,...,A_b] where A_i is the coefficient of
# z^i in the Laurent expansion of a matrix A whose entries are rational functions.

def devmat(A,a,b):
    m = A.rows
    L = [zeros(m,m) for k in range(b+1-a)]   
    for i in range(m):
        for j in range(m):
            coeffs = devserie(A[i,j],a,b)
            for k in range(b+1-a):
                L[k][i,j] = coeffs[k]
    return L


########################################################################################################
# SECTION : Ramification order
#
# We compute the ramification order d(A) of the p-Mahler system of matrix A with the function ramif_order(A,p).
# The functions of this section (except ramif_order) were given in the implemented algorithm of [FP22].

# Let P be a polynomial. Thanks to Cauchy's theorem, max_mod(P) returns an integer which is bigger that the modulus of the roots of P.

def max_mod(P):
    g = Poly(P,z).all_coeffs() #list of the coefficients of P, the first one is the coefficient of z^n, n=deg(P) 
    if len(g) == 1:
        return 1
    h = [abs(x) for x in g[1:]]
    k = math.ceil(1+max(h)/abs(g[0]))
    return k

# eval_pt(A) returns an integer z_0 greater than or equal to 2 such that the matrices A(z0),..., A(z0^(p^(m-2))) are well-defined and invertible.

def eval_pt(A):
    m = len(A[:,0])
    determ = A.det()
    P = numer(cancel(determ))
    L = [2, max_mod(P)]
    for i in range(m):
        for j in range(m):
            Pij = denom(cancel(A[i,j]))
            L.append(max_mod(Pij))
    z_0 = max(L)
    return z_0

# Let X=[x_1,...,x_n] and Y=[y_1,...,y_n], x_i,y_i are algebraic numbers, interpol(X,Y) returns a polynomial f such that f(x_i)=y_i. 

def interpol(X,Y):
    n = len(X)
    E = Matrix([Y])
    F = ones(n,1)
    for i in range(n):
        for j in list(range(i))+list(range(i+1,n)):
            F[i,0] = F[i,0]*(z-X[j])/(X[i]-X[j])
    G = E*F
    return G[0,0]

# Let z0 be an algebraic number, mat_values(A,p,z0) returns a matrix whose rows are e_1, e_2 A(z0)^(-1),..., e_m A(z0)^(-1)...A(z0^(p^(m-2)))
# where e_1, ..., e_m is the canonical basis of C^m.

def mat_values(A,p,z0):
    m = len(A[:,0])
    K = eye(m)
    L = eye(m)
    prod = eye(m)
    X = [z0**(p**j) for j in range(m-1)]
    for i in range(m-1):
        prod = prod*(A.subs(z,X[i])).inv()
        L[i+1,:] = K[i+1,:]*prod
    return L

# Let z0 = eval_pt(A).
# r1gauge(A,p,z0) returns the first row of a gauge transformation which transforms A into a companion matrix denoted by Acomp.
# gauge(A,p,z0) returns such a gauge transformation.
# main_cyclic_vector_algo(A,p) (see the beggining of the algorithm) returns a list made of the entries of the last row of Acomp.

def r1gauge(A,p,z0):
    m = len(A[:,0])
    X = [z0**(p**i) for i in range(m)]
    r = [interpol(X,list(mat_values(A,p,z0)[:,i])) for i in range(m)]
    return r

def gauge(A,p,z0):
    m = len(A[:,0])
    M = zeros(m)
    M[0,:] = Matrix([r1gauge(A,p,z0)])
    for i in range(1,m):
        N = list(((M[i-1,:]).subs(z,z**p))*A)
        O = [cancel(x) for x in N]
        M[i,:] = Matrix([O])
    return M

# main_cyclic_vector_algo(A,p) returns the last row of a companion matrix which is Q(z)-equivalent to the system (A,p).

def main_cyclic_vector_algo(A,p):
    m = len(A[:,0])
    z0 = eval_pt(A)
    P = gauge(A,p,z0)
    last_row_comp = list((P.subs(z,z**p)*A*(P.inv()))[m-1,:])
    return last_row_comp


# Returns the points defining the Newton polygon associated with the equation of a companion matrix equivalent to A
def pts_polyg(A,p):
    m = len(A[:,0])
    last_row_comp = main_cyclic_vector_algo(A,p)
    L1 = [p**i for i in range(m+1)]
    L2 = [valzero(x) for x in last_row_comp]+[0]
    return list(zip(L1,L2))


# Returns the points defining the Newton polygon associated with a linear Mahler equation
def pts_polyg_eq(L,p):
    m = len(L)
    L1 = [p**i for i in range(m+1)]
    L2 = [valzero(x) for x in L]+[0]
    return list(zip(L1,L2))
    

# If points:= pts_polyg(A,p) then lower_hull(points) returns the lower hull of the 
# Newton polygon associated with Acomp.
def lower_hull(points):
    points = sorted(set(points))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    add_pts = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: 
            if cross(lower[-2], lower[-1], p) == 0:
                add_pts.append(lower[-1])
            lower.pop()
        lower.append(p)
    return [lower, add_pts]

# Let E be the denominators of the slopes of the Newton polygon associated with Acomp.
# ramif_order(A,p) returns the least integer d relatively prime with p such that the denominators of 
# the elements of E divide dp^k for some k.

def ramif_order(A,p):
    L = pts_polyg(A,p)
    N = lower_hull(L)[0]
    n = len(N)
    numer_slopes = [N[i+1][1]-N[i][1] for i in range(n-1)]
    denom_slopes = [N[i+1][0]-N[i][0] for i in range(n-1)]
    slopes = list(zip(numer_slopes,denom_slopes))
    denom = [Fraction(x[0],x[1]).denominator for x in slopes]
    withoutp = []
    m = len(A[:,0]) # we know that the ramification order is smaller than p^m
    for x in denom:
        g = math.gcd(x,p**m)
        withoutp.append(x//g)
    return lcm(withoutp)


# Let E be the denominators of the slopes of the Newton polygon associated some equation L.
# ramif_order_eq(L,p) returns the least integer d relatively prime with p such that the denominators of 
# the elements of E divide dp^k for some k.
def ramif_order_eq(L,p):
    points = pts_polyg_eq(L,p)
    N = lower_hull(points)[0]
    n = len(N)
    numer_slopes = [N[i+1][1]-N[i][1] for i in range(n-1)]
    denom_slopes = [N[i+1][0]-N[i][0] for i in range(n-1)]
    slopes = list(zip(numer_slopes,denom_slopes))
    denom = [Fraction(x[0],x[1]).denominator for x in slopes]
    withoutp = []
    m = len(L) # we know that the ramification order is smaller than p^m
    for x in denom:
        g = math.gcd(x,p**m)
        withoutp.append(x//g)
    return lcm(withoutp)

########################################################################################################

# In all the sections below, A and p are respectively the matrix and the integer greater than 1 of the system ; 
# B is the inverse of A and m is the size of A (A and B are mxm matrices).

##########################################################################################################

# SECTION : compute the useful objects (S'_p; the integers nu, mu, ... ; the matrices M_l)

# integers(A,p,B,m) returns a list containing the valuation of A, the valuation of the inverse of A,
# the integer nuTheta, then the list S'_p, and the integers nuP, nu and mu.  

def integers(A,p,B,m):# B = A.inv() 
    vA = val(A)
    vdA = valzero(det(A))
    vB = val(B)
    L = [vA,vB]
    l = math.ceil((p*m*vA-p*vdA)/(p-1))
    if l == 0:
        nuTheta = 0
    else:
        if l % p == 0:
            nuTheta = l+1
        else:
            nuTheta = l
    L.append(nuTheta)
    Lint = list(range(nuTheta,1))
    n = math.floor(-nuTheta/p)
    for i in range(n):
        Lint.remove(-p*(i+1))
    L.append(Lint)
    nuP = math.ceil(vA/(p-1))
    L.append(nuP)
    L.append(min(nuP, p*nuP+vB) + nuTheta) 
    L.append(max(math.ceil(-((vB+nuTheta)/(p-1))),-(m-1)*nuP+vdA/(p-1)))
    return L #list which contains, in this order, val(A), val(B), nuTheta, S'_p, nuP, nu, mu
    

# matM_l(A,p,B,m,l,integ) computes the matrix M_l with l in S'_p and integ = integers(A,p,B,m),
# it will be a square matrix of size m*(mu-nu+1).
def matM_l(A,p,B,m,l,integ):
    vB = integ[1]
    mu = integ[-1]
    nu = integ[-2]
    nuP = integ[4]  
    L = zeros(m*(mu-nu+1),m*(mu-nu+1))
    if vB+p*nuP+l>mu: #matrix Ml has zero blocks for the lines i<= vB+p*nuP+l-nu or j<= nuP-nu thus,
        return L # if vB+p*nuP+l-nu>mu-nu then it is the null matrix.
    devmatrix = devmat(B,vB+p*nuP-p*mu,mu-l-p*nuP+p)
    for i in range(vB+p*nuP+l-nu,mu-nu+1): #otherwise M_{i,j} = B_{i+nu-l-1-p(j+nu-1)}
        for j in range(nuP-nu,mu-nu+1):
            L[i*m:(i+1)*m,j*m:(j+1)*m] = devmatrix[i+nu-l-vB-p*(j+nu-mu+nuP)]
    return L

#The following function returns the list of matrices [M_nuTheta, ..., M_i, ... M_0] for all i in S'_p.
def matricesM(A,p,B,m):
    integ = integers(A,p,B,m)
    Spprime = integ[3]
    L = [matM_l(A,p,B,m,i,integ) for i in Spprime]
    return L

        
 
########################################################################################################

# SECTION : compute the vector spaces.   

# given a list X containing a basis of a vector space and M a list of matrices [M1,...,Mk], vspaceU(M,X)
# returns a basis of span(M_iX for 1<=i<=k). Thus, for M = matricesM(A,p,B,m), it returns a basis of span(M_kX for k in S'_p).
def vspaceU(M,X):
    l1 = len(X)
    l2 = len(M)
    L = [M[i]*X[k] for k in range(l1) for i in range(l2)]#gives M[0]*X[0], ..., M[l2-1]*X[0], M[0]*X[1],...
    V = Matrix.hstack(*L)
    return V.columnspace()

# Let V and W be the bases of two vector spaces Vvs and Wvs (given by a list of column matrices),
# inter(V,W) returns a list whose elements form a basis of the intersection of Vvs and Wvs.    
def inter(V,W):
    B = V+W
    m = len(V)
    M = Matrix.hstack(*B)
    K = M.nullspace()
    if len(K)==0:
        return []
    N = Matrix.hstack(*K)
    T = N[0:m,:]
    U = M[:,0:m]
    V = U*T
    return V.columnspace()

# Let M be a matrix and let F and U be two bases of vector spaces (given by a list of column matrices),
# let F1 = MF+U and F2 = M^{-1}(F+U) then intervsp(M,F,U) returns the vector space F inter F1 inter F2. 
def intervsp(M,F,U):
    F1 = vspaceU([M],F)+U
    V = inter(F,F1)
    W = (Matrix.hstack(*(F+U))).columnspace() #it remains to compute V inter M^{-1}(W)
    MV =  [M*x for x in V]
    B = MV+W
    m = len(MV)
    MatMVandW = Matrix.hstack(*B)
    MatMVandW
    K = MatMVandW.nullspace()
    if len(K)==0:
        return []
    N = Matrix.hstack(*K)
    T = N[0:m,:] 
    F2 = vspaceU([Matrix.hstack(*V)],T.columnspace())
    return inter(V,F2)

# Let V be a vector subspace of W, then suppl(V,W) computes a basis of a supplementary of V inside W.
# (The vector spaces V and W are given by a list of column matrices which form a basis.)
def suppl(V,W):
    MatVandW = Matrix.hstack(*(V+W))
    basis = MatVandW.columnspace()# columnspace returns the independent columns in the same order they appear in the matrix. 
    l = len(V)
    return basis[l:]

from sympy import Matrix

def suppl_eye(V, W):
    """Computes the supplementary space of V when W is the identity matrix. It takes the normal form of the transpose of V, and checks for the first non-zero columns"""
    if not V:
        return W
    
    n = len(W)
    V_mat = Matrix.hstack(*V)
    
    # Transpose V
    V_T = V_mat.T
    
    
    _, pivot_cols = V_T.rref() 
    
    # pivot_cols contains the indices of the pivot columns of the transpose of V
    # = the indices of the pivot rows of V
    pivot_rows = set(pivot_cols)
    # The vectors e_i for which i is not in pivot_rows form a supplementary space
    non_pivot_rows = [i for i in range(n) if i not in pivot_rows]
    
    return [W[i] for i in non_pivot_rows]


# The following function returns a list containing the vector space E and U, given by a list of vectors which form a basis, of the article (see Algorithm 1),
# the matrix M=M0, the list of matrices Mk and the dimension of E.
def vspaces(A,p,B,m):
    integ = integers(A,p,B,m)
    mu = integ[-1]
    nu = integ[-2]
    nuP = integ[4] 
    X = [zeros(m*(mu-nu+1),1)]
    E = []
    U = []
    listM = matricesM(A,p,B,m)
    M = listM[-1]
    while len(X)<m:
        vspU = vspaceU(listM,X)
        identity = eye(m*(mu-nu+1))
        Fmatrix = identity[:,m*(nuP-nu):m*(mu-nu+1)]
        F = Fmatrix.columnspace()
        G = intervsp(M,F,vspU)
        while len(F)!=len(G):
            F = G
            G = intervsp(M,F,vspU)
        Y = suppl(X,inter(vspU,F))
        Z = suppl(Matrix.hstack(*(X+Y)).columnspace(),F)
        E.append(Matrix.hstack(*(Y+Z)))
        U.append(vspU)
        X = F
    return [E,U,M,listM,len(E)]

########################################################################################################

# SECTION : compute the matrices Theta and P.   

# Let U be a nonzero vector space (given by a list of column matrices which form a basis of U).
# matrixKer(U) returns a matrix K such that ker(K) = U.
def matrixKer(U):
    l = len(U)
    n = U[0].rows
    ident = (eye(n)).columnspace()
    basisUcompl = suppl_eye(U,ident) # will keep the vectors in U and complete it in a basis
    P = Matrix.hstack(*(U+basisUcompl))
    D = eye(n)
    D[:,0:l] = zeros(n,l)
    return P*D*(P.inv())
    

# Let M be a matrix, RedForm returns a list of three matrices P,Q,R and a number r such that R is a reduced matrix (Id_r & 0 \\ 0 & 0) and 
# PMQ = R.
def RedForm(M):
    l = M.rows
    m = M.cols
    B2 = M.nullspace()
    n = len(B2)
    ident = (eye(m)).columnspace()
    B1 = suppl_eye(B2,ident)
    Q = Matrix.hstack(*(B1+B2))
    X = Matrix.hstack(*B1)
    C1 = (M*X).columnspace()
    identit = (eye(l)).columnspace()
    C2 = suppl_eye(C1,identit)
    P = (Matrix.hstack(*(C1+C2))).inv()
    return [P,Q,P*M*Q,m-n]

    
# The following function returns the list of the matrices Theta_j (the diagonal blocks of Theta) 
# when integ = integers(A,p,B,m) and vsp = vspaces(A,p,B,m).
def diagTheta(A,p,B,m,integ,vsp):  
    mu = integ[-1]
    nu = integ[-2]
    nuP = integ[4] 
    r = vsp[-1]
    U = vsp[1]
    M = vsp[2]
    E = vsp[0]
    Theta = []
    for j in range(r):
        sizeTh = E[j].cols
        Uj = U[j] # a basis of the vector space U_{j-1}
        if len(Uj) == 0:
            K = eye(m*(mu-nu+1))
        else:
            K = matrixKer(Uj) #K is a matrix whose kernel is Uj so we have KEj = KMEjThetaj. Obj : find a Thetaj
        bigM = K*M*(E[j])
        red = RedForm(bigM)
        P = red[0]
        r = red[-1] # rank of bigM
        N = P*K*E[j]
        QinvTheta = zeros(sizeTh)
        QinvTheta[:r,:] = N[:r,:] #now, we have to complete QinvTheta to have an invertible matrix
        identit = eye(sizeTh-r) #we complete it with the columns of the identity matrix
        reduced, pivots = QinvTheta.echelon_form(simplify=simplify, with_pivots=True) 
        for i in range(sizeTh):
            l = 0
            if i not in pivots:
                QinvTheta[r:,i] = identit[:,l]
                l+=1 
        Theta.append(red[1]*QinvTheta)
    return Theta 
    
# If M is a matrix, then rcef(M) returns the reduced columns echelon form of M.
def rcef(M):
    C = (M.T).rref()
    return C[0].T
            
# The following function returns the matrix Theta of Algorithm 1 when integ = integers(A,p,B,m) and vsp = vspaces(A,p,B,m).
def matTheta(A,p,B,m,integ,vsp):
    ThetaDiag = diagTheta(A,p,B,m,integ,vsp)
    r = len(ThetaDiag)
    E = vsp[0]
    listM = vsp[3]
    l = len(listM)#card of S'_p
    M = vsp[2]
    mu = integ[-1]
    nu = integ[-2]
    Spprime = integ[3]
    N = zeros(m*(mu-nu+1),0)
    Theta = zeros(m,m)
    sizeEj = []
    for j in range(r):#put the diagonal matrices
        lj = E[j].cols 
        Theta[sum(sizeEj):sum(sizeEj)+lj,sum(sizeEj):sum(sizeEj)+lj] = ThetaDiag[j]
        sizeEj.append(lj)
    for j in range(1,r):
        Ej = E[j] 
        lj = Ej.cols 
        Thetaj = ThetaDiag[j]
        Rj = Ej-M*Ej*Thetaj
        Nsup = zeros(m*(mu-nu+1),0)
        for k in range(l):     
            for i in range(j):
                Nsup = Nsup.col_insert(0,listM[l-1-k]*E[j-1-i])
        #Thus we want to find a X such that Nsup*X = Rj (exists by construction but not unique)
        N = Nsup.col_insert(0,-Rj) # it is (-Rj|MnuTheteE1|...|MnuThetaE_{j-1}|...|M0E1|...|M0E_{j-1})        
        B = Matrix.hstack(*(N.nullspace()))
        X = rcef(B)
        lX = X.rows
        Nj = sum(sizeEj[0:j])
        ThetaBlocColj = X[lX-Nj:lX,0:lj]
        for x in range(1,l):
            ThetaBlocColj = ThetaBlocColj + z**(Spprime[l-1-x])*(X[lX-Nj*(x+1):lX-Nj*x,0:lj])
        Theta[0:Nj,Nj:Nj+lj]= ThetaBlocColj
    return Theta

# The following function returns the matrix Pbar of Algorithm 1 when integ = integers(A,p,B,m) and vsp = vspaces(A,p,B,m).        
def matPbar(A,p,B,m,integ,vsp):   
    E = Matrix.hstack(*vsp[0])
    nuP = integ[-3]
    nu = integ[-2]
    mu = integ[-1]
    P = zeros(m,m)
    for i in range(mu-nuP+1):
        P = P + E[(i+nuP-nu)*m:(i+nuP-nu+1)*m,:]*z**(nuP+i)
    return P
        

###########################################################################
###########################################################################
# EXAMPLES :
    
    
############################EXAMPLE 1######################################   
# Companion matrix associated with the Mahler equation (2.1) of the article [CDDM18]:
# "Computing solutions of linear Mahler equations" of Chyzak, Dreyfus, Dumas, and Mezzarobba in Math. Comp., 87 :2977–3021, 2018. 
# (Also in section 5.2 of the article [FP22] : "An algorithm to recognize regular singular Mahler systems" of Faverjon and Poulet in Math. Comp., 91 :2905-2928, 2022.)
# A=Matrix([[0,1], [-z**6*(1+z)*(1-z**21-z**30)/(z**3*(1-z**3+z**6)*(1-z**7-z**10)), (1-z**28-z**31-z**37-z**40)/(z**3*(1-z**3+z**6)*(1-z**7-z**10))]])

# Algo1(A,3) returns [2,
# Matrix([
# [3*z**18 - 2*z**16 + 2*z**14 - 2*z**12 + z**10 - z**8 + z**6, z**19 - z**17 + z**15 - z**13 + z**11 - z**9 + z**7 - z**5 + z**3 - z + 1/z],
# [z**18, -z**15 + z**9 - z**3 + z**(-3)]]),
# Matrix([
# [1, 0],
# [0, 1]])]

# Thus the ramification order is 2, a matrix Theta is Matrix([
#  [1, 0],
#  [0, 1]]) which is a constant matrix, so the system is regular singular and Puiseux-equivalent to the identity matrix.
#  We have Pbar = Matrix([
#  [3*z**18 - 2*z**16 + 2*z**14 - 2*z**12 + z**10 - z**8 + z**6, z**19 - z**17 + z**15 - z**13 + z**11 - z**9 + z**7 - z**5 + z**3 - z + 1/z],
#  [z**18, -z**15 + z**9 - z**3 + z**(-3)]])].


# We recover the constant matrix given by the main algorithm in [FP22]. The matrix Pbar is the truncation of the gauge transform Psi(z**2) obtained thanks to this algorithm.
    
##############################EXAMPLE 2################################################           
# Rudin-Shapiro example from Section 5.3 of "An algorithm to recognize regular singular Mahler systems" of Faverjon and Poulet in Math. Comp., 91 :2905-2928, 2022.
# B = Matrix([[1,z],[1,-z]]) 
# Algo1(B,2) returns 
# [1,
# Matrix([
# [4*z**2, -4*z**2 + 2*z - 2],
# [  -4*z, -2*z**2 + 6*z - 2]]),
# Matrix([
# [-2, 4 - 1/z],
# [ 0,       1]])]


#############################EXAMPLE 3#################################################
# Baum-Sweet example: C = Matrix([[0,1],[1,-z]])
# Algo1(C,2) returns 
# [1,
# Matrix([
# [1, 0],
# [0, 1]]),
# Matrix([
# [0, 1],
# [1, 0]])]
 
 

##############################EXAMPLE 4######################################
# Direct sum of Rudin-Shapiro and Baum-Sweet
# D = Matrix([[1,z,0,0],[1,-z,0,0],[0,0,0,1],[0,0,1,-z]])
# Algo1(D,2) returns
#[1,
# Matrix([
# [0, 4*z**2,    0, -4*z**2 + 2*z - 2],
# [0,   -4*z,    0, -2*z**2 + 6*z - 2],
# [z,      0,    1,                 0],
# [1,      0, z**2,                 0]])
#Matrix([
# [0,  0, 1,       0],
# [0, -2, 0, 4 - 1/z],
# [1,  0, 0,       0],
# [0,  0, 0,       1]])]


###############################EXAMPLE 5##############################
# A caracteristic 0 analog to the example with Carlitz's Zeta function
# E = Matrix([[pi-z**3,pi-z**3],[0,1]])
# Algo1(E,3) returns 
# [1,
# Matrix([
# [1, 0],
# [0, 1]]),
# Matrix([
# [pi, pi],
# [ 0,  1]])]

################################EXAMPLE 6################################
# The first example given in Section 8 of the paper [FP25]: "Regular Singular Mahler Equations and Newton Polygons", J. Math. Soc. Japan, 1: 1--33 (2025). 
# F=Matrix([[0,1],[(-1-z)/z**8,(z**2+z**3+z**7)/z**8]])
# Algo1(F,2) returns
#[1,
# Matrix([
# [-5*z**15/2 + 17*z**14/2 - 5*z**13/2 - 3*z**12/2 + z**11 + 3*z**10/2 - 3*z**9/2 - 3*z**8/2 - z**7 + 4*z**6 - 3*z**5/2 - z**4/2 - z**3/2 + 3*z**2/2 - z - 1/(2*z) + z**(-2) - 1/(2*z**3), 7*z**15/2 - 29*z**14/2 + 9*z**13/2 + 7*z**12/2 - 2*z**11 - 7*z**10/2 + 5*z**9/2 + 7*z**8/2 + 2*z**7 - 8*z**6 + 7*z**5/2 - z**4/2 + 5*z**3/2 - 7*z**2/2 + 2*z - 1 + 1/(2*z) + 1/(2*z**3)],
# [-3*z**14/2 + 6*z**12 - 5*z**10/2 - 3*z**6/2 + 5*z**4/2 - 3*z**2/2 + 1/2 - 1/(2*z**2) + 1/(2*z**4) - 1/(2*z**6),                                                                   5*z**14/2 - 10*z**12 + 9*z**10/2 - z**8 + 7*z**6/2 - 9*z**4/2 + 5*z**2/2 - 3/2 + 1/(2*z**2) + 1/(2*z**4) + 1/(2*z**6)]]),
# Matrix([
# [ 1/2, 1/2],
# [-1/2, 3/2]])]

################################EXAMPLE 7#####################################
# The inverse system of Example 6
# G = Matrix([[0,1],[-z**8/(1+z),(z**2+z**3+z**7)/(1+z)]])
# Algo1(G,2) returns
# [1,
# Matrix([
# [   z, z**23 + z**22/2 + 3*z**21/2 + z**20/2 + z**19 + z**18/2 + z**17 + z**16/2 + z**15 + z**14/2 + z**13 + z**12/2 + z**11 + z**10/2 + z**9/2 + z**8/2 + z**7/2 + z**6/2],
# [z**2, z**22 + z**20/2 + z**18/2 + z**16/2 + z**14/2 + z**12/2]]),
# Matrix([
# [1, 0],
# [0, 1]])]


################################EXAMPLE 8#####################################
# The analog of Example 6 with a coefficient 5
# H=Matrix([[0,1],[(-1-z)/z**8,(z**2+z**3+5*z**7)/z**8]])
# Algo1(H,2) returns
# [1,
# Matrix([
# [-z**15/6 - 179*z**14/6 + 139*z**13/6 + 53*z**12/6 - 13*z**11/3 - 49*z**10/6 + 7*z**9/2 + 161*z**8/30 + 13*z**7/15 - 104*z**6/15 + 161*z**5/30 - 29*z**4/6 + 167*z**3/30 - 49*z**2/30 + 11*z/15 - 14/15 - 1/(30*z) + z**(-2) - 1/(30*z**3), 7*z**15/6 - 25*z**14/6 + 17*z**13/6 + 7*z**12/6 - 2*z**11/3 - 11*z**10/6 + 3*z**9/2 + 19*z**8/30 + 2*z**7/15 - 16*z**6/15 + 19*z**5/30 - z**4/6 + 13*z**3/30 - 11*z**2/30 + 4*z/15 - 1/15 + 1/(30*z) + 1/(30*z**3)],
# [5*z**14/6 - 20*z**12/3 + 31*z**10/6 - 14*z**8/3 + 161*z**6/30 - 47*z**4/30 + 7*z**2/10 - 9/10 - 1/(30*z**2) + 29/(30*z**4) - 1/(30*z**6),                                                                                    z**14/6 - 4*z**12/3 + 5*z**10/6 - z**8/3 + 19*z**6/30 - 13*z**4/30 + 3*z**2/10 - 1/10 + 1/(30*z**2) + 1/(30*z**4) + 1/(30*z**6)]]),
# Matrix([
# [29/30,  1/30],
# [-1/30, 31/30]])]

################################EXAMPLE 9#####################################
# The analog of Example 7 with a coefficient 5
# J= Matrix([[0,1],[-z**8/(1+z),(z**2+z**3+5*z**7)/(1+z)]])
# Algo1(J,2) returns
# [1,
# Matrix([
# [z**23 + z**22/6 + 35*z**21/6 + 5*z**20/6 + z**19 + z**18/6 + 5*z**17/3 + 5*z**16/6 + z**15 + z**14/6 + z**13 + z**12/6 + z**11 + z**10/6 + z**9/6 + z**8/6 + z**7/6 + z**6/6, -185*z**23/6 - 145*z**22/36 - 3677*z**21/36 - 821*z**20/36 - 51*z**19/2 - 193*z**18/36 - 755*z**17/18 - 335*z**16/36 - 20*z**15 - 145*z**14/36 - 23*z**13 - 169*z**12/36 - 12*z**11 - 121*z**10/36 - 181*z**9/36 - 91*z**8/36 - 103*z**7/36 - 43*z**6/36 - z**5/3 - z**4/6 - z/24],
# [z**22 + z**20/6 + z**18/6 + z**16/6 + z**14/6 + z**12/6,                                                                                                                                    -16*z**22 - 145*z**20/36 - 205*z**18/36 + 5*z**17/6 - 115*z**16/36 - 127*z**14/36 - 67*z**12/36 - z**10/3 + z**9/6 - z**8/6 + z**7/6 - z**2/24]]),
# Matrix([
# [1, -4 - 1/z + z**(-5)],
# [0,                  1]])]
# 

################################EXAMPLE 10#####################################
# The caracteristic sequence of Cantor triadic integers is associated to the 2-Mahler system with matrix
# K=￼Matrix([[(4*z + 1)/(3*z**2 + 6*z + 3), (z - 2)/(z**2 + 2*z + 1)], [-z/(3*z**2 + 6*z + 3), 3/(3*z**2 + 6*z + 3)]])
# Algo1(K,2) returns
#[1,
# Matrix([
# [1, 0],
# [0, 1]]),
# Matrix([
# [1/3, -2],
# [  0,  1]])]

################################EXAMPLE 11#####################################
#Let a_n denote the number of subword of length n in the Thue-Morse sequence. Its generating series satisfies the 2-Mahler equation with coefficients
# L=[2*z**9-2*z**8+2*z**6-2*z**4+z**3+z**2-z,-2*z**11-2*z**10-2*z**9+4*z**8-4*z**7-4*z**6+3*z**5+3*z**4-6*z**3+z**2+z+1,4*z**13-4*z**12+8*z**11-4*z**10+4*z**9+4*z**7-4*z**6+8*z**5-5*z**4+8*z**3-6*z**2+4*z-3,-(4*z**11-4*z**10+4*z**9-2*z**8+8*z**7-8*z**6+8*z**5-4*z**4+4*z**3-4*z**2+4*z-2)]
# AlgoEq(L,2) returns 
# [Matrix([[0,1,0],
# [0,0,1],
# [(-2*z**9 + 2*z**8 - 2*z**6 + 2*z**4 - z**3 - z**2 + z)/(-4*z**11 + 4*z**10 - 4*z**9 + 2*z**8 - 8*z**7 + 8*z**6 - 8*z**5 + 4*z**4 - 4*z**3 + 4*z**2 - 4*z + 2), (2*z**11 + 2*z**10 + 2*z**9 - 4*z**8 + 4*z**7 + 4*z**6 - 3*z**5 - 3*z**4 + 6*z**3 - z**2 - z - 1)/(-4*z**11 + 4*z**10 - 4*z**9 + 2*z**8 - 8*z**7 + 8*z**6 - 8*z**5 + 4*z**4 - 4*z**3 + 4*z**2 - 4*z + 2), (-4*z**13 + 4*z**12 - 8*z**11 + 4*z**10 - 4*z**9 - 4*z**7 + 4*z**6 - 8*z**5 + 5*z**4 - 8*z**3 + 6*z**2 - 4*z + 3)/(-4*z**11 + 4*z**10 - 4*z**9 + 2*z**8 - 8*z**7 + 8*z**6 - 8*z**5 + 4*z**4 - 4*z**3 + 4*z**2 - 4*z + 2)]]),
# 1,
# Matrix([
# [-3*z**2/2 - z/2 + 1/2, z**2 - 1, 3*z**2/2 + 3*z/2 + 3/2],
# [1/4 - 3*z**2/4,       -1,         7*z**2/4 + 7/4],
# [1/8,       -1,                   15/8]]),
# Matrix([
# [  0, 0,   1],
# [  -1, 1,   1],
# [-1/2, 0, 3/2]])]


###################################EXAMPLE 12#######################################
# The sequence of Euclidian matching with parameters a=b=1 introduced in Example 88 of 
# [P. Dumas, Récurrences mahlériennes, suites automatiques, études asymptotiques, Thèse, Université de Bordeaux I, Talence (1993).] 
# is related to the 2-Mahler system with matrix
# M=Matrix([[(z**3-2*z**2-z)/(z**3+z**2-z-1),(z**3+z)/(z**2+2*z+1),(-z**4+z**3+z**2+z)/(z**3+z**2-z-1)],[(z**3+z)/(z**3+z**2-z-1),(-z**2+z)/(z**2+2*z+1),-2*z**2/(z**3+z**2-z-1)],[-2*z**2/(z**3+z**2-z-1),2*z/(z**2+2*z+1),(z**3+z)/(z**3+z**2-z-1)]])
# Algo1(M,2) returns
# [3,
# Matrix([
# [     4*z**9/5 + 2*z**6/5, 5*z**10/6 + z**7/2 + z**4/6, 5*z**11/6 + z**8/2 + z**5/6],
# [z**9 + 3*z**6/5 + z**3/5, 5*z**10/6 + z**7/2 + z**4/6,   z**11 + 2*z**8/3 + z**5/3],
# [z**9 + 3*z**6/5 + z**3/5,   z**10 + 2*z**7/3 + z**4/3,   z**11 + 2*z**8/3 + z**5/3]]),
# Matrix([
# [1, 0, 0],
# [0, 0, 1],
# [0, 1, 0]])]

##################################EXAMPLE 13#########################################
# The sequence of valuation 5-adic of n! is associated with the 5-Mahler system whose matrix is
# N=Matrix([[(26*z**4 + 21*z**3 + 16*z**2 + 11*z + 6)/(5*z**8 + 10*z**7 + 15*z**6 + 20*z**5 + 25*z**4 + 20*z**3 + 15*z**2 + 10*z + 5), (-25*z**4 - 19*z**3 - 13*z**2 - 7*z - 1)/(5*z**8 + 10*z**7 + 15*z**6 + 20*z**5 + 25*z**4 + 20*z**3 + 15*z**2 + 10*z + 5), (4*z**4 + 3*z**3 + 2*z**2 + z)/(5*z**8 + 10*z**7 + 15*z**6 + 20*z**5 + 25*z**4 + 20*z**3 + 15*z**2 + 10*z + 5)], [1/(z**4 + z**3 + z**2 + z + 1), 0, 0], [(-15*z**4 - 10*z**3 - 5*z**2 + 5)/(z**8 + 2*z**7 + 3*z**6 + 4*z**5 + 5*z**4 + 4*z**3 + 3*z**2 + 2*z + 1), (19*z**4 + 13*z**3 + 7*z**2 + z - 5)/(z**8 + 2*z**7 + 3*z**6 + 4*z**5 + 5*z**4 + 4*z**3 + 3*z**2 + 2*z + 1), (-3*z**4 - 2*z**3 - z**2 + 1)/(z**8 + 2*z**7 + 3*z**6 + 4*z**5 + 5*z**4 + 4*z**3 + 3*z**2 + 2*z + 1)]])
# Algo1(N,5) returns
# [1,
# Matrix([
# [1, 0, 0],
# [0, 1, 0],
# [0, 0, 1]]),
# Matrix([
# [6/5, -1/5, 0],
# [  1,    0, 0],
# [  5,   -5, 1]])]


###################################EXAMPLE 14#######################################
# The 2-Mahler system associated with the sequence a_n = 1 + integer part of log_2(n) has matrix
# O=Matrix([[z, 1, z],[-1+z, 2, 2*z],[0, 0, 1]])
# Algo1(0,2) returns
# [1,
# Matrix([
# [1, 0, 0],
# [0, 1, 0],
# [0, 0, 1]]),
# Matrix([
# [ 0, 1, 0],
# [-1, 2, 0],
# [ 0, 0, 1]])]


##################################EXAMPLE 15##########################################
# The 2-Mahler system associated with the sequence of sum of digits in the Gray code representation  of an integer has matrix
# P=Matrix([[(z**2+z+2)/(z**3+z**2+z+1),-1/(z**2+1)],[2/(z**3+z**2+z+1),(z-1)/(z**2+1)]])
# Algo1(P,2) returns
# [1,
# Matrix([
# [z + 2,     z**2 - 1],
# [    2, z**2 + z - 1]]),
# Matrix([
# [1, 0],
# [0, 1]])]


#################################EXAMPLE 16#########################################
# The Hanoi Tower morphism is associated with a 2-Mahler system. Its matrix is 
# Q=Matrix(
#[[0, 0, 0, 1/(z**3 + 1), -z/(z**3 + 1), z**2/(z**3 + 1)],
#[0, 0, 0, -z/(z**3 + 1), z**2/(z**3 + 1), 1/(z**3 + 1)],
#[0, 0, 0, z**7/(z**8 + z**5), z**5/(z**8 + z**5), -z/(z**3 + 1)],
#[0, 0, 1/(z), 1/(z**3 + 1), -z/(z**3 + 1), -1/(z**4 + z)],
#[0, 1/(z), 0, -z/(z**3 + 1), -1/(z**4 + z), z**5/(z**8 + z**5)],
#[1/(z), 0, 0, -1/(z**4 + z), 1/(z**3 + 1), -z/(z**3 + 1)]])
# Algo1_with_d(Q,2,1) returns
# [Matrix([
# [z**5 + z**2,    z**3 + 1,    z**4 + z,          z**3 + 1,          z**4 + z,       z**5 + z**2],
# [   z**4 + z, z**5 + z**2,    z**3 + 1,       z**5 + z**2,          z**3 + 1,          z**4 + z],
# [   z**3 + 1,    z**4 + z, z**5 + z**2,          z**4 + z,       z**5 + z**2,          z**3 + 1],
# [z**5 + z**2,    z**3 + 1,    z**4 + z,          z**3 + 1,          z**4 + z, z**5 + z**2 + 1/z],
# [   z**4 + z, z**5 + z**2,    z**3 + 1, z**5 + z**2 + 1/z,          z**3 + 1,          z**4 + z],
# [   z**3 + 1,    z**4 + z, z**5 + z**2,          z**4 + z, z**5 + z**2 + 1/z,          z**3 + 1]]),
# Matrix([
# [0, 0, 1, 1/z,   1,   0],
# [0, 1, 0,   1,   0, 1/z],
# [1, 0, 0,   0, 1/z,   1],
# [0, 0, 0,  -1,   0,   0],
# [0, 0, 0,   0,   0,  -1],
# [0, 0, 0,   0,  -1,   0]]),]