import numpy as np 

def TestLineFun1(x):
    return x**4-14*(x**3)+60*(x**2)-70*x
    # x* = 0.780885825794867, f(x*) = -24.369601567258060

def TestLineFun2(x):
    return (0.65-0.75/(1+x**2))-0.65*x*np.arctan2(1,x)
    # x* = 0.4808678353168805, f(x*) = -0.310020501948328

def TestLineFun3(x):
    return -(108*x-x**3)/4
    # x* = 6, f(x*) = -108