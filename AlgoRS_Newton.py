# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# This is an implementation of the Algorithm 2 in the paper "regular singular Mahler equations and Newton polygons" of Faverjon and Poulet.
# In what follows, we use the same notations introduced in this paper.

# Let L be a p-Mahler operator with p > 1 an integer. 
# algo_eq_rs(L,p) returns whether or not the Mahler equation Ly = 0 is regular singular at 0.
 
# Remark : if L:=am M^m+...+ a1M + a0 with M the p-Mahler operator, we represent it by a list [a0,...,am] (see some
# examples at the end). 


#########################################################################################################
#Python 3.9
from sympy import *
z = symbols('z')
X = symbols('X')# X represents the indeterminate lambda of the paper
y = symbols('y')# it represents the "lambda-c" where c is an exponent attached to a slope of the Newton polygon 
from fractions import *
from functools import reduce
from math import *


    
def valzero(f): # compute the valuation at 0 of a rational function f.
    irred = cancel(f) 
    num = numer(irred)
    den = denom(irred)
    if f == 0:
        return float("inf") 
    if num.subs(z,0) == 0:    
        g = (Poly(num,z).EM()).as_expr()
        return int(degree(g, gen=z))
    if den.subs(z,0) == 0:    
        g = (Poly(den,z).EM()).as_expr()
        return int(- degree(g, gen=z))
    else:
        return 0



def devserie(f,a,b): # a<=b integers, returns the list [x_a,...,x_b] where x_i is the coefficient of z^i
# in the Laurent expansion of a rational function f.
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


#################################################################################
#################################################################################

# We are interested in the Mahler equation Ly = 0 where L:= am M^m+...+ a1M + a0 
# with M the p-Mahler operator. Here it is represented by a list [a0,...,am].  

    

def pts_polyg(L,p): #list of the points used to construct the lower hull of the Newton polygon
    m = len(L)
    L1 = [p**i for i in range(m)]
    L2 = [valzero(x) for x in L]
    return list(zip(L1,L2))

def insert_sort(L1,L2): #L1=[a1,a2,...,ak] and L2 two lists of integers with L1 sorted ; returns a list of 
# sorted lists of the form [ai, ..., a{i+1} ] with elements of L2 between ai and a{i+1} (if any). 
    L = [[L1[i],L1[i+1]] for i in range(len(L1)-1)]
    for x in L2:
        for i in range(len(L)):
            if x>L[i][0] and x<L[i][-1]:
                L[i].append(x)
                L[i] = sorted(L[i])           
                break
    return L

def slopes_multiplicities(L,p): # it returns the slopes, their multiplicities and the numbers used to 
# construct the characteristic polynomials
    points = pts_polyg(L,p) 
    n = len(points)
    ptsNumero = list(zip(points,range(n))) 
    def cross(onum, anum, bnum): 
        o = onum[0]
        a = anum[0]
        b = bnum[0]
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]) 
    lower = [] 
    add_num = [] 
    for i in range(n):
        pt = ptsNumero[i]
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0: 
            if cross(lower[-2], lower[-1], pt) == 0:
                add_num.append(lower[-1][1])             
            lower.pop() 
        lower.append(pt)             
    n = len(lower)
    numeros = list(zip(*lower))[1] 
    multiplicities = [numeros[i+1] - numeros[i] for i in range(n-1)]
    lowerpts = list(zip(*lower))[0] 
    numer_slopes = [lowerpts[i+1][1]-lowerpts[i][1] for i in range(n-1)]
    denom_slopes = [lowerpts[i+1][0]-lowerpts[i][0] for i in range(n-1)]
    NumDen = list(zip(numer_slopes,denom_slopes))
    slopes = [Fraction(x[0],x[1]) for x in NumDen] 
    numpolycharac = insert_sort(numeros,add_num)
    return [slopes,multiplicities,numpolycharac]  
    
    


def poly_charac(L,S,p):  # S = slopes_multiplicities(L,p)
# it returns the list of the characteristic polynomials divided by a power of X (to avoid the zero roots)
    l=S[2]
    n = len(l)
    chi = []
    for i in range(n):
        coeff = l[i]
        f = [(Poly(L[m],z).coeffs()[-1])*X**(m-coeff[0]) for m in coeff]
        chi.append(sum(f))
    return chi


def factor_multiplicities(L,S,p): # same entries as before
# it returns a matrix M whose coefficient (i,j) is the multiplicity of the i-th irreducible factor of 
# the j-th characteristic polynomial.
    listcharac = poly_charac(L,S,p)
    factors_with_repetitions = [factor_list(x,domain=QQ)[1] for x in listcharac]
    n = len(listcharac)
    factors = []
    for i in range(n):
        factors_i = factors_with_repetitions[i]
        new_factor = [x[0] for x in factors_i if x[0] not in factors]
        factors += new_factor 
    m = len(factors) 
    M = zeros(m,n)
    num_factors = list(zip(factors, range(m)))
    for i in range(n):
        factors_i = factors_with_repetitions[i]
        for j in range(len(factors_i)):
            factor = factors_i[j]
            numero = [k[1] for k in num_factors if k[0] == factor[0]]#cela va donner un seul numéro
            M[numero[0], i] = factor[1]
    return factors, M
        
    
def cld_solution(L,p,j,k,S,F): #j is the number of the slope (begin with j=0), k is the number of an exponent c
# it returns r mod(X-c)^{mcj+scj} (remark : we don't take the constant coefficient of r here : prod ci,k / a_{0,val(a0)})
    multiplicities = S[1]
    factor_only = [x for x in F[0]]
    factor_matrix = F[1]
    mcj_list = list(factor_matrix[k,:])
    if j == 0:
        sum_ri = 0
        scj = 0
    else:
        sum_ri = sum(multiplicities[0:j])
        scj = sum(mcj_list[0:j])
    mult_factors_j = list(factor_matrix[:,j])       
    mcj = mult_factors_j[k]               
    del(mult_factors_j[k])                            
    n = len(mult_factors_j)
    k_factor = factor_only[k]
    del(factor_only[k])
    other_factors = [factor_only[x] for x in range(n) if mult_factors_j[x] != 0]
    other_factors_multiplicities = [x for x in mult_factors_j if x != 0]
    m = len(other_factors)
    c = rootof(k_factor,0)
    other_exponents = [rootof(x,0) for x in other_factors]
    elts = [(y+c-other_exponents[x])**(other_factors_multiplicities[x]) for x in range(m)]
    product = prod(elts)
    f = 1/(product*(y+c)**sum_ri)
    devserie = series(f, y,0, mcj).removeO()
    return [devserie*(y**scj), scj+mcj,c]



def MahlerEqXc(L,p,j,k,S,F):
# it returns a matrix whose entry (i,j) is the coefficient of M^j of L_i, which is the coefficient of (X-c)^i in L_lambda
    n = len(L)
    sol = cld_solution(L,p,j,k,S,F)
    scj_mcj = sol[1]
    c = sol[2] 
    M = zeros(scj_mcj, n)
    P=[L[i]*y**i for i in range(n)] # it is L_lambda
    for i in range(n):
        h=Poly(P[i].subs(y,y+c),y).all_coeffs()
        h.reverse()
        M[:,i] = h[:scj_mcj]+ list(zeros(1,max(scj_mcj-len(h),0)))
    return M


def subs_z_To_zd(L,d): # it transforms a list of functions f(z) into a list of f(z^d)
    Lpowerd = []
    for x in L:
        if type(x) == int or type(x) == float or type(x) == complex:
            Lpowerd.append(x)
        else:
            Lpowerd.append(x.subs(z,z**d)) 
    return Lpowerd
    

def L_X_f_mod(p,M,f): # M =  MahlerEqXc(L,p,j,k,S,F) ; f a polynomial in y=X-c (mod y^{scj+mcj})
# it returns a function g = L_X(f) mod (X-c)^{scj+mcj}.
    l = shape(M)
    h = Poly(f,y).all_coeffs()
    h.reverse()    
    list_end = h + list(zeros(1,1+l[0]-len(h)))
    N = zeros(l[0],l[1])
    s = []
    for i in range(l[0]):
        indices = list(range(i+1))
        indices.reverse()
        for j in range(l[1]):
            eltsM = [M[x,j] for x in indices]
            eltsh = [list_end[x] for x in range(i+1)]
            eltshpower = subs_z_To_zd(eltsh,p**j) 
            multiply = [eltsM[k]*eltshpower[k] for k in range(i+1)]
            N[i,j] = sum(multiply)
        s.append(expand(sum(N[i,:]))*y**i)
    return sum(s)
     
    
def pi_val_g(L,p,g):  
# it returns the number pi(valz(g)) with g(y,z) a function.
    h = Poly(y*g,y).all_coeffs()  
    listval = [valzero(x) for x in h]
    val = min(listval) 
    if val == float("inf"):
        return float("inf")
    else:
        listslopes = [(val-valzero(L[i]))/(p**i) for i in range(len(L))]
        return Fraction(max(listslopes))


def cldz(f):
# it returns cld_z(f) with f(y,z) a function.
    if f == 0:
        return float("inf")
    d = -valzero(f)
    cld = series(f*z**d,z,0,1).removeO()
    return cld
    

def slopes_denom_lcm(L,S,p):
# it returns false if there is one non-admissible slope or a denominator of a slope which is not coprime with p 
# else, it returns the lcm of the denominators of the slopes, denoted by d.
    slopes = S[0]
    denom_slopes = [denom(x) for x in slopes]
    n = len(slopes)
    for i in range(n):
        if denom_slopes[i]%p == 0:
            return False
    return lcm(*denom_slopes)



def listvaly(L): # it returns the list of the valuations of functions in a list L with respect to the variable y.
    n = len(L)
    val = [0]*n
    for i in range(n):
        f=L[i].subs(y,X)
        g=f.subs(z,y)
        h=g.subs(X,z)
        val[i]=valzero(h)
    return val


def solutionMod(a,b,l,m):
# it returns, if it exists, a polynomial h(y) of degree <=m such that ah = b modulo y^l.
# else, it returns false. 
    L = [a,b]
    val = listvaly(L)
    if val[1] < val[0]:
        return False
    else:
        h = series(cancel(b/a),y,0, l-val[0]).removeO()
        coeffs = Poly(h,y).all_coeffs()
        if len(coeffs)>m+1:
            return False
        else:
            return h


    
def algo_eq_rs(L,p): # main algo : it returns whether or not the p-Mahler equation Ly = 0 is regular singular at 0.
    valuations=[valzero(x) for x in L]
    maxval=max(valuations)
    if p>=maxval:
        p=max(2,maxval)
    S=slopes_multiplicities(L,p)
    if len(S[0])==1:     
        return True
    d= slopes_denom_lcm(L,S,p)
    if d == False:                  
        return False      
    Lnoramif = subs_z_To_zd(L,d)    
    S=[[d*x for x in S[0]], S[1], S[2]]
    F=factor_multiplicities(Lnoramif,S,p)
    m = len(S[0])
    n = len(F[0])
    for j in range(m):
        num_exp = [] 
        for k in range(n):
            if (F[1])[k,j]!=0:
                num_exp.append(k)
    #    print(num_exp)
        for k in num_exp:
            r_and_power = cld_solution(Lnoramif, p, j, k, S, F)
            r = r_and_power[0]
            power = r_and_power[1]
         #   print(r)
            M = MahlerEqXc(Lnoramif, p, j, k, S, F)
            f = simplify(r*(z**(-S[0][j])))
            g = simplify(L_X_f_mod(p,M,f))
            v = pi_val_g(Lnoramif,p,g)
          #  print([r,M,f,g,v])
            if v == float("inf"):
                return "True1"
            while v <= -S[0][0]:
            #    print(v)
                den = denom(v)
           #     print(k,v,'denom v=',den)
                if gcd(den,p)!= 1 or den >p**(len(L)-1)-1:
                    return "False1"
                a = cldz(L_X_f_mod(p,M,z**v))
            #    print('a=',a)
                if a == 0:
                    return "False2"
                b = cldz(g)
             #   print('b=',b)
                valya = listvaly([a])
                h = solutionMod(a, b, power, power-1-valya[0])
            #    print('h=',h)
                if h == False:
                    return "False3"
                f = simplify(f-h*z**v)
             #   print('f=',f)
                g = simplify(L_X_f_mod(p,M,f))
                v = pi_val_g(Lnoramif,p,g)
                if v == float("inf"):
                    return "True2"
    return True 
                


###########################################################################
###########################################################################
# EXAMPLES :
    
    
############################EXAMPLE 1 ###################################### 

# L1 = [z, -1-z, 1]
# p = 2
# S = slopes_multiplicities(L1,p)
# chi = poly_charac(L1,S,p)
# algo_eq_rs(L1,p)

# We have S = [[Fraction(-1, 1), Fraction(0, 1)], [1, 1], [[0, 1], [1, 2]]] that is
# two slopes [Fraction(-1, 1), Fraction(0, 1)] of multiplicity [1, 1] respectively. 

# Also, chi = [1 - X, X - 1], the two polynomials whose roots are the exponents. 

# algo_eq_rs(L1,p) returns 'True1', the equation is regular singular at 0. 


############################EXAMPLE 2 ###################################### 

# L2 = [z, -1-2*z, 1]
# p = 2
# S = slopes_multiplicities(L2,p)
# chi = poly_charac(L2,S,p)
# algo_eq_rs(L2,p)

# Its Newton polygon and the exponents attached to each slope are the same than the ones of
# the first example. We have the same S and chi as before but here

# algo_eq_rs(L2,p) returns 'False1', the equation is not regular singular at 0. 


############################EXAMPLE 3 ###################################### 

# L3 = [2*z**3, 1-z, -2+z**2, 1+z, 2*z**2]
# p = 2
# S = slopes_multiplicities(L3,p)
# chi = poly_charac(L3,S,p)
# multiplicity = factor_multiplicities(L3,S,p)
# algo_eq_rs(L3,p)

# We have S = [[Fraction(-3, 1), Fraction(0, 1), Fraction(1, 4)],[1, 2, 1], [[0, 1], [1, 2, 3], [3, 4]]] 
# that is three slopes -3, 0, 1/4  of multiplicity 1, 2, 1 respectively. 

# Also, chi = [X + 2, X**2 - 2*X + 1, 2*X + 1], the three polynomials whose roots are the exponents. 

# multiplicity = ([X + 2, X - 1, 2*X + 1],
 # Matrix([
 # [1, 0, 0],
 # [0, 2, 0],
 # [0, 0, 1]])), this means that the irreducible factors of the characteristic polynomials are [X + 2, X - 1, 2*X + 1]
 # the multiplicity of X+2 is [1,0,0] that is 1 (resp. 0 and 0) for the first (resp. second, third) characteristic polynomial etc.

# algo_eq_rs(L3,p) returns 'False', the equation is not regular singular at 0. 



############################EXAMPLE 4 ###################################### 

# L4 = [1+z,-z**2-z**3-z**7,z**8]
# p = 2
# S = slopes_multiplicities(L4,p)
# chi = poly_charac(L4,S,p)
# multiplicity = factor_multiplicities(L4,S,p)
# algo_eq_rs(L4,p)

# We have S =  [[Fraction(2, 1), Fraction(3, 1)], [1, 1], [[0, 1], [1, 2]]]
# chi = [1 - X, X - 1]
# multiplicity = ([X - 1], Matrix([[1, 1]])), that is, there is only one irreducible factor X-1, 
# and the associated exponent (which is 1) is of multiplicity 1 for the two characteristic polynomial. 

# algo_eq_rs(L4,p) returns that the equation is regular singular at 0 (it returns True). 
# In details, the is only one exponent (numbered "0"). Thus num_exp = [0], for the first iteration,
# the reduction r of r1,1(λ) modulo (λ−1) up to a multiplicative constant is r = 1 and we have
# [f, g] = [ z**(-2), 1 - z**3] up to a constant and v =  Fraction(0, 1) with the notations of the paper.
# Since v = 0 > -mu1 = -2, the first iteration of "for j in range(m)" finishes and we start the second one 
# which gives :  num_exp = [0] (only one exponent of this Mahler equation, which is numbered 0).
#  We have r1,2(λ) modulo (λ − 1)²  = y, which is "lambda - c" with c=1 (the exponent). Then, 
# [f, g] = [y/z**3, -y*z + y/z**2] and v = Fraction(-2, 1). We have v<= -mu1 = -2 and v in 1/d*Z so 
# it uses the loop "while" which gives : the denominator of v is 1, alpha = -y (which is 1 - lambda), 
# b= y (which is lambda - 1) and h= -1 (using the notations of the article). Then, it replaces
# f by (y + z)/z**3 = (lambda-1)/z**3 + 1/z**2, g by (-y*(z*(z**3 + z - 2) + 1) + z*(1 - z**3))/z and v by -1.
# Now, we have v = -1 > -mu1 = -2, the second and last iteration of "for j in range(m)" finishes and the 
# algorithm returns True. 


############################EXAMPLE 5 ###################################### 
# The inverse of the equation of Example 4, which is L5=[z**8, -z**7 - z**3 - z**2, z + 1], is also regular singular at 0. 
# algo_eq_rs(L5,2) returns True1.

############################EXAMPLE 6 ###################################### 

# L6=[1+z,-z**2-z**3-2*z**7,z**8] with p=2 is regular singular but its inverse equation is not :
# algo_eq_rs(L7,2) returns False1 with L7=[z**8, -2*z**7 - z**3 - z**2, z + 1].