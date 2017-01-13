import numpy
from numpy import pi, cos, sin
from scipy.special import hyp2f1, gamma, factorial

def mehler(m, p, s, theta):

    # mehler conical function K of order m and degree n = -1/2 + i*p*pi/S
    # S is a conical surface parameter S = log(b/a)

    n_imag = p*pi/s
    
    n_factor = gamma(0.5+m+1j*n_imag)/gamma(0.5-m+1j*n_imag)

    return (-1)**m/(2**m*factorial(m))*n_factor*sin(theta)**m*\
        hypergeom(m, p, s, theta)

def hypergeom(m, p, s, theta, iter_max=1000):
    a = 1
    hyp = 0
    n_imag = p*pi/s
    
    for i in range(1, iter_max+1):
        a = ((m+1/2+i)**2 + n_imag**2)/(i+m+1)/(i+1)*a
        hyp += a*sin(theta/2)**(2*i)

    return hyp
