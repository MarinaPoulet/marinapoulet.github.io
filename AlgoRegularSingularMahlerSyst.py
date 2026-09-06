# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# This is an implementation of the Algorithm 3 in the paper 
# "An algorithm to recognize regular singular Mahler systems" of Faverjon and Poulet.
# A Mahler system is a pair (A,p) where A is a matrix whose entries are rational functions and p is an integer.
# mainalgo(A,p,n) returns whether or not the Mahler system (A,p) is regular singular at 0.
# If so, it returns a constant matrix which is Puiseux equivalent to the system (A,p) and the truncation at
# order n of the Puiseux development of an associated gauge transformation.

# In the mean time, we implemented an algorithm related to the Cyclic Vector Lemma.
# main_cyclic_vector_algo(A,p) corresponds to Algorithm 1. It returns the last row of a companion matrix which is Q(z)-equivalent to the system (A,p). 

# In what follows, we use the notations introduced in the paper "An algorithm to recognize regular singular Mahler systems" 
# of Faverjon and Poulet (the sets D, D0 ; the matrices M_d, N_d, Psi (a gauge transformation), E (the matrix whose columns are
# a basis of the vector space X of Algorithm 3 ; the integers nu_d, mu_d etc).

def mainalgo(A,p,n):
    #L=choice_d(A,p)          #Test every d in D.
    L = denom_slopes(A,p)    #Compute first a d in D0 and then tests only this precise d.
    for i in range(len(L)):
        d = L[i]
        test = AlgoRS_d_grdd(A,p,d)
        if test[0] == True: 
            E = test[1]
            Ainv = test[2]
            nu_d = test[3]
            mu_d = test[4]
            R = matcst(Ainv,p,d,E,nu_d,mu_d)
#L[i] = d, the ramification ; test[1] = E
            Psi = gaugetr(A,Ainv,p,d,E,R,n,nu_d,mu_d)
            return ["the system is regular singular", R, Psi] 
    return "the system is not regular singular"  

def main_cyclic_vector_algo(A,p):
    m = len(A[:,0])
    z0 = eval_pt(A)
    P = gauge(A,p,z0)
    last_row_comp = list((P.subs(z,z**p)*A*(P.inv()))[m-1,:])
    return last_row_comp


#########################################################################################################
from sympy import *
z = symbols('z')

from fractions import *
from functools import reduce

import math


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

 
# Let X be a list of column matrices of same size, matricify(X) returns a matrix
# whose rows are the column matrices of X. 

def matricify(X):
    m = len(X[0])
    n = len(X)
    M = zeros(m,n)
    for i in range(n):
        M[i] = X[i]
    return M


# Let V and W be the bases of two vector spaces Vvs and Wvs (given by a list of column matrices),
# inter(V,W) returns a list whose elements form a basis of the intersection of Vvs and Wvs.    

def inter(V,W):
    B = V+W
    m = len(V)
    M = matricify(B)
    K = M.nullspace()
    if len(K)==0:
        return []
    N = matricify(K)
    T = N[0:m,:]
    U = M[:,0:m]
    V = U*T
    return V.columnspace()
 
    
# A d-gridded matrix M is represented by a list of d+1 elements made of the d blocks (which can be blocks with 0 row or/and 0 columns) 
# of the d-gridded matrix (the first block is the block constructed thanks to the matrices which can be nonzero in the first row of M
# and the rows 1 + d*n, n an integer, etc) and the permutation associated with M (which is the last element of the list).    
    
# blockmat_grdd(A^(-1),p,nu_d,mu_d,d,v0) returns the matrix M_d of Algorithm 3.

def blockmat_grdd(Ainv,p,nu_d,mu_d,d,v0):    
    J = [j for i in range(d) for j in range(d) if (nu_d+i-p*(j+nu_d)) % d == 0] #the permutation 
    m = Ainv.rows
    devmatrix = devmat(Ainv,v0,max(v0,int((mu_d-p*nu_d)/d)))
    r = int((mu_d-nu_d+1)/d)  
    L = [] #construction of the blocks defining the d-gridded matrix
    for i in range(d):
        if J[i] >= (mu_d-nu_d+1)-d*r:  
            s = r
        else:
            s = r+1
        if i >= (mu_d-nu_d+1)-d*r:
            t = r
        else:
            t = r+1
        M = zeros(t*m,s*m) #the block containing the matrices which can be nonzero in the lines i+1, i+1+d, i+1+2d etc
        c = int((nu_d+i-p*(J[i]+nu_d))/d)
        for l in range(s):
            for k in range(t):
                if c+k-p*l < v0:
                    M[k*m:(k+1)*m,l*m:(l+1)*m]=zeros(m,m)
                else:
                    M[k*m:(k+1)*m,l*m:(l+1)*m]=devmatrix[c+k-p*l-v0]
        L.append(M)
    L.append(J) 
    return L

# matconditions(A^(-1),p,nu_d,mu_d,d,v0) returns the matrix N_d of Algorithm 3.
    
def matconditions_grdd(Ainv,p,nu_d,mu_d,d,v0):    
    m = Ainv.rows
    if nu_d == mu_d or (((1-p)*nu_d-1)/d) < v0:
        r = int((mu_d-nu_d+1)/d) 
        J = [i for i in range(d)] #we can choose the identity permutation in this case
        L = []
        for i in range(d):
            if J[i] >= (mu_d-nu_d+1)-d*r:
                s = r
            else:
                s = r+1
            L.append(zeros(m*s,m*s))
        L.append(J)
        return L    
    devmatrix = devmat(Ainv,v0,int(((1-p)*nu_d-1)/d))
    start = d*v0+p*nu_d
    J = [j for i in range(d) for j in range(d) if int(start+i-p*(j+nu_d)) % d == 0] #the permutation
    r1 = int((nu_d-start)/d)  #(number of blocks in the columns of N_d)/d
    r2 = int((mu_d-nu_d+1)/d) #(number of blocks in the rows of N_d)/d
    L = []
    for i in range(d):
        if J[i] >= (mu_d-nu_d+1)-d*r2:
            s2 = r2
        else:
            s2 = r2+1
        if i >= nu_d-start - d*r1:
            s1 = r1
        else:
            s1 = r1+1
        N = zeros(s1*m,s2*m)
        c = int((start+i-p*(J[i]+nu_d))/d) #(start+i-p*(J[i]+nu_d))/d is an integer, "int" is just here to have type(c)=int
        for k in range(s1):
            for l in range(s2):
                if c+k-p*l < v0: 
                    N[k*m:(k+1)*m,l*m:(l+1)*m] = zeros(m,m)
                else:
                    N[k*m:(k+1)*m,l*m:(l+1)*m] = devmatrix[c+k-p*l-v0]
        L.append (N)
    L.append(J)
    return L


# Let D and E be two d-gridded matrices, multiply_grdd(D,E,d) returns D*E if it is well-defined.
 
def multiply_grdd(D,E,d):
    sigmaD = D[-1]
    sigmaE = E[-1]
    J = [sigmaE[sigmaD[i]] for i in range(d)]
    L = [(D[i])*(E[sigmaD[i]]) for i in range(d)]
    L.append(J)
    return L


# Let J be a permutation, inversion_permutation(J) returns the inverse of J.

def inversion_permutation(J):
    d = len(J)
    L = [j for i in range(d) for j in range(d) if J[j] == i]
    return L


# Let E be a d-gridded matrix with blocks of size m, kernel_grdd(E,d,m) returns the kernel of E in a d-gridded form. 

def kernel_grdd(E,d,m):
    J = inversion_permutation(E[-1]) #the permutaton of the kernel is the inverse of the permutation associated with E.
    ker = []
    for i in range(d):
        nullspaces = (E[J[i]]).nullspace()
        remainder = len(nullspaces) % m
        if remainder == 0:
            if nullspaces == []:
                M = zeros((E[J[i]]).cols,m)
            else:
                M = matricify(nullspaces)
        else:
            Mcols = nullspaces + (m-remainder)*[zeros((E[J[i]]).cols,1)]        
            M = matricify(Mcols)
        ker.append(M)
    ker.append(J)
    return ker   


# Let V and W be two vector spaces spanned by the columns of two d-gridded matrices (with blocks of size m) D, E respectively.
# inter_grdd(D,E,m) returns the intersection of V and W in a d-gridded matrix with the permutation replaced by the dimension of the 
# vector space, that is a list made of the matrices of the d-gridded form and the dimension of this intersection which is the last 
# element of this list.

def inter_grdd(D,E,m):
    n = len(E)-1
    L = []
    nber = 0 #nber will be the dimension of the intersection of the two vector spaces
    for i in range(n):
        if D[i].rows == 0:
                L.append(zeros(D[i].rows,m))
        else:
            if (D[i]).columnspace() == [] or (E[i]).columnspace() == []:
                M = [zeros(D[i].rows,1)]
            else:
                M = inter((D[i]).columnspace(), (E[i]).columnspace())
                nber = nber + len(M) 
            if M == []:
                M = [zeros(D[i].rows,1)] 
            remainder = len(M) % m
            if  remainder == 0:
                L.append(matricify(M))
            else:
                Mcols = M + (m-remainder)*[zeros((D[i]).rows,1)]
                L.append(matricify(Mcols))          
    return L+[nber]
 

# Let A be a matrix whose entries are rational functions, let p be an integer greater than or equal to 2 and let d be a positive integer.
# AlgoRS_d(A,p,d) is an algorithm which returns whether or not the system Y(z^p)=A(z)Y(z) is equivalent to a constant system
# with a gauge transformation whose entries are Puiseux series with ramification d.
# If it is the case, it returns "True" and the basis of the vector space X_d (of Algorithm 3).
           
def AlgoRS_d_grdd(A,p,d):
    if A.det() == 0:
        return [False, "the matrix is not invertible"]
    m = A.rows
    Ainv = A.inv()
    v0=val(Ainv)
    nu_d = math.ceil(d*val(A)/(p-1))
    mu_d = math.ceil(-d*v0/(p-1))
    M = blockmat_grdd(Ainv,p,nu_d,mu_d,d,v0)
    N = matconditions_grdd(Ainv,p,nu_d,mu_d,d,v0)
    K = kernel_grdd(N,d,m)    
    c_d = m*(mu_d-nu_d+1)
    t = math.ceil(log(c_d,2))
    J = K
    Mpower = [M]
    for i in range(1,t+1):
        Mpower.append(multiply_grdd(Mpower[i-1],Mpower[i-1],d)) #Mpower = [M,M²,M⁴,...,M^(2^t)]
    for i in range(1,t+1):
        J = inter_grdd(J,multiply_grdd(Mpower[t-i],J,d),m) #J = { x such that x \in J and M^{2^{t-i}}x \in J}
        Inter = J
        J = J[:-1] 
        J.append([i for i in range(d)]) 
        if Inter[-1] < m:
            return [False, "the system is not regular singular"]           
    X_dminus = Inter[:-1] + [[i for i in range(d)]]
    X_d = multiply_grdd(Mpower[t], X_dminus, d)
    Inter = inter_grdd(X_d,X_d,m)
    dim = Inter[-1]
    if dim < m:
        return [False, "the system is not regular singular"]
#We have X_d in a d-gridded form, in what follows, we transform it in a "normal form"        
    vsp_grdd = Inter[:-1]      
    n = len(vsp_grdd)
    vsp = []
    for i in range(n):
        V = (vsp_grdd[i]).columnspace()
        if V != []:
            vsp.append(matricify(V))
    q = sum([(vsp_grdd[i]).rows for i in range(n)]) 
    InterFinal = zeros(q,m)
    l = 0
    v = len(vsp)
    for i in range(v):
        for k in range(int(vsp[i].rows/m)):
            InterFinal[(i+k*n)*m:(i+k*n+1)*m,l:l+(vsp[i]).cols] = (vsp[i])[k*m:(k+1)*m,:]      
        l = l + (vsp[i]).cols
    return [True, InterFinal,Ainv,nu_d,mu_d]


    
######################################################################################
######################################################################################
# We consider the Mahler system Y(z^p)=A(z)Y(z) with A an invertible matrix whose entries are rational functions and p an 
# integer greater than or equal to 2. Let m be the size of A.
# Now, we want to test AlgoRS_d_grdd(A,p,d) with an admissible ramification d.
# There are two possibilities :
# Let D be the set of integers n such that n <= p^m-1 and gcd(n,p)=1.
    # POSSIBILITY 1) run AlgoRS_d_grdd(A,p,d) with all the biggest integers of the set D with respect to the partial order "divisibility"
# Let D_0 be the subset of D such that if d is in D_0, a Mahler system is regular singular at 0 if and only if
# it is equivalent to a constant matrix with an associated gauge transformation whose ramification is d.
    # POSSIBILITY 2) use the cyclic vector lemma to find an element of D_0 and run AlgoRS_d_grdd(A,p,d) with this d.   

#######################################################################################
#POSSIBILITY 1) choice_d(A,p) returns the biggest integers of the set D with respect to the partial order "divisibility". 

def choice_d(A,p):
    m = A.rows
    L = list()
    t = True
    if p**m-1 == 1:
        return [1]
    for i in range(p**m-1,1,-1):
        t = True    
        if math.gcd(i,p) == 1:
                for j in range(1,len(L)):
                       if i in divisors(L[j-1]):
                           t = False
                if t == True:
                    L.append(i)
    return L

##########################################################################################
#POSSIBILITY 2) The cyclic vector.

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

# companion(A,p) returns a companion matrix associated with A.

# def companion(A,p):
#     m = len(A[:,0])
#     z0 = eval_pt(A)
#     P = gauge(A,p,z0)
#     Acomp = P.subs(z,z**p)*A*(P.inv())
#     Acomp_irred = zeros(m)
#     for i in range(m):
#         for j in range(m):
#             Acomp_irred[i,j] = cancel(Acomp[i,j])
#     return Acomp_irred
    

# If points:= pts_polyg(A,p) then lower_hull(points) returns the lower hull of the 
# Newton polygon associated with Acomp.

def pts_polyg(A,p):
    m = len(A[:,0])
    last_row_comp = main_cyclic_vector_algo(A,p)
    L1 = [p**i for i in range(m+1)]
    L2 = [valzero(x) for x in last_row_comp]+[0]
    return list(zip(L1,L2))
    
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
# denom_slopes(A,p) returns the lcm of the elements of E which are coprime to p.

def denom_slopes(A,p):
    L = pts_polyg(A,p)
    N = lower_hull(L)[0]
    n = len(N)
    numer_slopes = [N[i+1][1]-N[i][1] for i in range(n-1)]
    denom_slopes = [N[i+1][0]-N[i][0] for i in range(n-1)]
    slopes = list(zip(numer_slopes,denom_slopes))
    denom = [Fraction(x[0],x[1]).denominator for x in slopes]
    admissible = [x for x in denom if math.gcd(x,p)==1]
    return [lcm(admissible)]
#denom_slopes(A,.p) is an element of D_0.


###########################################################################################
###########################################################################################


# The following algorithm returns if the Mahler system Y(z^p)=A(z)Y(z) is regular singular at 0 and, 
# if it is the case, it returns the ramification d of an associated gauge transformation.

def AlgoRS(A,p):
    #L = choice_d(A,p)       #If you want to use POSSIBILITY 1)
    L = denom_slopes(A,p)    #If you want to use POSSIBILITY 2)
    for i in range(len(L)):
        test = AlgoRS_d_grdd(A,p,L[i])
        if test[0] == True: 
            return ["the system is regular singular",L[i]]
#L[i]=d, the ramification
    return "the system is not regular singular"

########################################################################################
#######################################################################################
# If one wants to compute a constant matrix and a gauge transformation 
# associated with the system, if it is regular singular,
# one needs the following functions.


# If the Mahler system Y(z^p)=A(z)Y(z) is regular singular at 0 then 
# matcst(Ainv,p,d,E,nu_d,mu_d) returns a constant matrix associated with A where Ainv=A^{-1}.


def reconstM(M,d,m):
    J=M[-1]
    L=M[:-1]
    for i in range(len(L)):
        t=t+(L[i]).rows
    return t
          

def matcst(Ainv,p,d,E,nu_d,mu_d):
    m = Ainv.rows
    B = Ainv.subs(z,z**d)
    
    L = zeros(m*(mu_d-nu_d+1),m*(mu_d-nu_d+1))
    devmatrix = devmat(B,nu_d-p*mu_d,mu_d-p*nu_d)
    for i in range(mu_d-nu_d+1):
        for j in range(mu_d-nu_d+1):
            L[i*m:(i+1)*m,j*m:(j+1)*m] = devmatrix[i-p*(j+nu_d-mu_d)]
    M = L #the matrix M_d of Algorithm 3 (which is not given in a d-gridded form)
    
    R = zeros(m)
    n = m*(mu_d-nu_d+1)
    Matr = zeros(n,m+1)
    Matr[:,1:] = E
    T = zeros(n,m+1)
    for i in range(E.cols):
        T[:,0] = M*E[:,i]
        N1 = (T+Matr).nullspace()
        N2 = N1[0] #donne une matrice colonne
        N2norm = 1/N2[0,0]*N2       
        coeff = - N2norm[1:,0]
        R[:,i] = coeff
    return R

# Let d be a positive integer and let B=A^(-1)(z^d). 
# pair(n,vA,vAinv,p,d) returns [k,l] such that k+pl=n with k greater than or equal to the valuation of B at 0
# and l at least nu_d where vA is the valuation of A and vAinv is the valuation of A^(-1).

def pair(n,vA,vAinv,p,d):
    L = []
    for k in range(d*vAinv,n-p*math.ceil(d*vA/(p-1))+1):
        if (n-k)%p == 0:
            L.append([k,int((n-k)/p)])
    return L
    

# If the Mahler system Y(z^p)=A(z)Y(z) is regular singular at 0 then we denote by Psi an associated gauge transformation and d its ramification.
# gaugetr(A,Ainv,p,d,E,matcst(A,p,d,E),n,nu_d,mu_d) returns the coefficients of Psi(z^d) from z^(nu_d) (nu_d being a lower bound of the valuation at 0 of Psi) to z^{nu_d+n-1}. 

def gaugetr(A,Ainv,p,d,E,R,n,nu_d,mu_d):
    m = A.rows      
    L = [E[i*m:(i+1)*m,:] for i in range(int(E.rows/m))]
    Lz = [z**i for i in range(nu_d,mu_d+1)] 
    vA = val(A)    
    vAinv = val(Ainv)      
    if n <= mu_d-nu_d+1:
        multLzL = [x*y for x,y in zip(L[:n],Lz[:n])] 
        thesum = zeros(m)
        for i in range(n):
            thesum = thesum + multLzL[i]  
        return thesum
    for k in range(mu_d+1, nu_d+n):
        indices = pair(k,vA,vAinv,p,d)
        indices_k = [(indices[i])[0]-(indices[0])[0] for i in range(len(indices))]
        indices_l = [(indices[i])[1]-nu_d for i in range(len(indices))]
        devB = devmat(Ainv.subs(z,z**d), (indices[0])[0], (indices[-1])[0])
        listBk = [devB[i] for i in indices_k]
        listEl = [L[i] for i in indices_l]
        prodBkEl = ([x*y for x,y in zip(listBk,listEl)])
        thesum = zeros(m)
        for i in range(len(prodBkEl)):
            thesum = thesum + prodBkEl[i]
        Ek = (thesum)*(R**(-1)) 
        L.append(Ek)
        Lz.append(z**k)
    lastmult = [x*y for x,y in zip(L,Lz)]
    thesum = zeros(m)
    for i in range(len(lastmult)):
        thesum = thesum + lastmult[i]
    return thesum



###########################################################################
###########################################################################
# EXAMPLES :
    
    
############################EXAMPLE 1######################################   
# Companion matrix associated with the Mahler equation of section 5.2, which also is Equation (2.1) of the article [CDDM18]:
# "Computing solutions of linear Mahler equations" of Chyzak, Dreyfus, Dumas, and Mezzarobba in Math. Comp., 87 :2977–3021, 2018. 
 
# A=Matrix([[0,1], [-z**6*(1+z)*(1-z**21-z**30)/(z**3*(1-z**3+z**6)*(1-z**7-z**10)), (1-z**28-z**31-z**37-z**40)/(z**3*(1-z**3+z**6)*(1-z**7-z**10))]])

# AlgoRS(A,3) returns ['the system is regular singular', 2] so the system is regular singular with an associated gauge transformation that has ramification d=2.

# nu_d = math.ceil(d*val(A)/(p-1))=-3 (d=2 and p=3)
# mu_d = math.ceil(-d*val(A.inv())/(p-1))=6

# The matrix M_d of Algorithm 3 is M_d=Matrix([
# [-1, -3,  0,  0,  1,  2,  0,  0, -1, -1,  0,  0,  1,  0,  0,  0,  0, 0, 0, 0],
# [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0],
# [ 0,  0,  1,  2,  0,  0, -1, -1,  0,  0,  1,  0,  0,  0,  0,  0,  0, 0, 0, 0],
# [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0],
# [ 1,  4,  0,  0, -1, -2,  0,  0,  1,  1,  0,  0, -1,  0,  0,  0,  0, 0, 0, 0],
# [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0],
# [ 0,  0, -1, -3,  0,  0,  1,  2,  0,  0, -1, -1,  0,  0,  1,  0,  0, 0, 0, 0],
# [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0],
# [-1, -4,  0,  0,  1,  2,  0,  0, -1, -1,  0,  0,  1,  0,  0,  0,  0, 0, 0, 0],
# [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0],
# [ 0,  0,  1,  4,  0,  0, -1, -2,  0,  0,  1,  1,  0,  0, -1,  0,  0, 0, 0, 0],
# [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0],
# [ 1,  4,  0,  0, -1, -3,  0,  0,  1,  2,  0,  0, -1, -1,  0,  0,  1, 0, 0, 0],
# [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0],
# [ 0,  0, -1, -4,  0,  0,  1,  2,  0,  0, -1, -1,  0,  0,  1,  0,  0, 0, 0, 0],
# [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0],
# [-1, -4,  0,  0,  1,  4,  0,  0, -1, -2,  0,  0,  1,  1,  0,  0, -1, 0, 0, 0],
# [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0],
# [ 0,  0,  1,  4,  0,  0, -1, -3,  0,  0,  1,  2,  0,  0, -1, -1,  0, 0, 1, 0],
# [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0, 0, 0, 0]])
# and its d-gridded form is : blockmat_grdd(A**(-1),p,nu_d,mu_d,d)=
# [Matrix([
#  [-1, -3,  1,  2, -1, -1,  1,  0,  0, 0],
#  [ 0,  0,  1,  0,  0,  0,  0,  0,  0, 0],
#  [ 1,  4, -1, -2,  1,  1, -1,  0,  0, 0],
#  [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 0],
#  [-1, -4,  1,  2, -1, -1,  1,  0,  0, 0],
#  [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 0],
#  [ 1,  4, -1, -3,  1,  2, -1, -1,  1, 0],
#  [ 0,  0,  0,  0,  1,  0,  0,  0,  0, 0],
#  [-1, -4,  1,  4, -1, -2,  1,  1, -1, 0],
#  [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 0]]),
#  Matrix([
#  [ 1,  2, -1, -1,  1,  0,  0,  0, 0, 0],
#  [ 0,  0,  0,  0,  0,  0,  0,  0, 0, 0],
#  [-1, -3,  1,  2, -1, -1,  1,  0, 0, 0],
#  [ 0,  0,  1,  0,  0,  0,  0,  0, 0, 0],
#  [ 1,  4, -1, -2,  1,  1, -1,  0, 0, 0],
#  [ 0,  0,  0,  0,  0,  0,  0,  0, 0, 0],
#  [-1, -4,  1,  2, -1, -1,  1,  0, 0, 0],
#  [ 0,  0,  0,  0,  0,  0,  0,  0, 0, 0],
#  [ 1,  4, -1, -3,  1,  2, -1, -1, 1, 0],
#  [ 0,  0,  0,  0,  1,  0,  0,  0, 0, 0]]),
#  [0, 1]]

# The d-gridded form of the matrix N_d of Algorithm 3 is : matconditions_grdd(A**(-1),p,nu_d,mu_d,d)=
# [Matrix([
#  [ 1,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [-1,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 1,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [-1, -1,  1,  0,  0, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 1,  1, -1,  0,  0, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [-1, -1,  1,  0,  0, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 1,  2, -1, -1,  1, 0, 0, 0, 0, 0],
#  [ 1,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [-1, -2,  1,  1, -1, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 1,  2, -1, -1,  1, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0]]),
#  Matrix([
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 1,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [-1,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 1,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [-1, -1,  1,  0,  0, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 1,  1, -1,  0,  0, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [-1, -1,  1,  0,  0, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [ 1,  2, -1, -1,  1, 0, 0, 0, 0, 0],
#  [ 1,  0,  0,  0,  0, 0, 0, 0, 0, 0],
#  [-1, -2,  1,  1, -1, 0, 0, 0, 0, 0],
#  [ 0,  0,  0,  0,  0, 0, 0, 0, 0, 0]]),
#  [0, 1]]

# The vector space X = X_2 of Algorithm 3 is spanned by AlgoRS_d_grdd(A,p,d)[1]=
# Matrix([
# [ 0,  0],
# [ 1,  0],
# [ 0,  0],
# [ 0,  0],
# [ 1,  0],
# [ 0,  0],
# [ 0,  0],
# [ 0,  0],
# [-1,  0],
# [ 0,  0],
# [ 0,  0],
# [ 0,  0],
# [ 1,  0],
# [-1,  0],
# [ 0,  0],
# [ 0,  0],
# [-1,  0],
# [ 0,  0],
# [ 0, -1],
# [ 0,  0]])

# mainalgo(A,p,20) returns
# ['the system is regular singular',
#  Matrix([
#  [1, 0],
#  [0, 1]]),
#  Matrix([
#  [z**15 - z**13 + z**11 - z**9 + z**7 - z**5 + z**3 - z + 1/z, 2*z**16 - 2*z**14 + 2*z**12 - z**10 + z**8 - z**6],
#  [-z**15 + z**9 - z**3 + z**(-3),  0]])]

# Thus, the system is regular singular and equivalent to the identity matrix.
# Let Psi be an associated gauge transformation.
# The last matrix is the Puiseux expansion of Psi(z^d)=Psi(z^2)
# between z^{nu_d}=z^{-3} and z^{nu_d+19}=z^{16}.
# The entries of the first line of this matrix are the solutions given in Section 2.1 of [CDDM18].
    
    
    
##############################EXAMPLE 2################################################           
# Rudin-Shapiro example from Section 5.3: B = Matrix([[1,z],[1,-z]]) 
# denom_slopes(B,2) returns [1], thus the integer 1 is an admissible ramification index.
# Then AlgoRS_d_grdd(B,2,1) returns [False, 'the system is not regular singular'].


#############################EXAMPLE 3#################################################
# Baum-Sweet example: C = Matrix([[0,1],[1,-z]])
# denom_slopes(C,2) returns [1], thus the integer 1 is an admissible ramification index.
# mainalgo(C,2,20) returns ['the system is regular singular',
 # Matrix([
 # [0, 1],
 # [1, 0]]),
 # Matrix([
 # [z**19 + z**16 + z**7 + z**4 + z, z**15 + z**12 + z**9 + z**3 + 1],
 # [               z**18 + z**6 + 1,             z**14 + z**8 + z**2]])]
# Thus, the system is regular singular at 0 and it is Q((z))-equivalent to the constant system 
# whose matrix is R = Matrix([[0, 1], [1, 0]]).
# The last matrix is the Puiseux expansion from z^{nu_d} to z^{nu_d+19} (here nu_d=0, d=1)
# of an associated gauge transformation.
# If ones iterates this system to obtain a 4-Mahler system with matrix D=C.subs(z,z**2)*C. It is 
# regular singular and mainalgo(D,4,20)[1] returns the identity matrix, that is R^2.


#############################EXAMPLE 4#################################################
# We go back to the matrix of the system of Example 1 and we now consider the inverse matrix F=A.inv().
# Then AlgoRS(F,3) returns 'the system is not regular singular'. 
# It gives an example of a regular singular system whose inverse system is not regular singular at 0.


#############################EXAMPLE 5#################################################
# Consider once again the matrix A of Example 1. We now consider it as a p-Mahler system with p=2,4,5,6,... 
# mainalgo(A,2,1) returns 'the system is not regular singular'.
# The function 
    # for i in range(4,101):
        # print('for p=', i, AlgoRS(A,i))
# gives: 
# for p= 4 the system is not regular singular
# for p= 5 the system is not regular singular
# for p= 6 the system is not regular singular
# for p= 7 the system is not regular singular
# .... and so on until
# for p= 100 the system is not regular singular