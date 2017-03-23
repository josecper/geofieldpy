import numpy
import mfunnorm
from numpy import pi, cos, sin
from scipy.special import hyp2f1, gamma, factorial

def mehler(m, p, s, theta):
    """ mehler conical functions """

    # mehler conical function K of order m and degree n = -1/2 + i*p*pi/S
    # S is a conical surface parameter S = log(b/a) k

    n_imag = p*pi/s
    
    n_factor = gamma(0.5+m+1j*n_imag)/gamma(0.5-m+1j*n_imag)

    return (-1)**m/(2**m*factorial(m))*n_factor*sin(theta)**m*\
        hypergeom(m, p, s, theta)

def hypergeom(m, p, s, theta, iter_max=100):
    """ hypergeometric function """
    a = 1
    hyp = 1
    n_imag = p*pi/s
    
    for i in range(0, iter_max):
        #hyp += a*sin(theta/2)**(2*i)
        a = ((m+1/2+i)**2 + n_imag**2)/(i+m+1)/(i+1)*a
        hyp += a*((1-cos(theta))/2)**(i+1)

    return hyp

def mehler_t(m, tau, theta, theta_0=numpy.pi, normalized=False, mfactor = False):
    """ mehler functions as calculated by thebault the great french man """
    theta = numpy.rad2deg(theta)
    theta_0 = numpy.rad2deg(theta_0)
    norm = int(normalized)
    uf = numpy.frompyfunc(lambda x:mfunnorm.mfunnorm(m,tau,x,theta_0, norm)[0], 1, 1)

    if mfactor:
        return (-1)**m*numpy.float64(uf(theta))
    else:
        return numpy.float64(uf(theta))

def dmehler_t(m, tau, theta, theta_0=numpy.pi, normalized=False, mfactor = False):
    """ mehler functions derivative as calculated by thebault the great french man """
    theta = numpy.rad2deg(theta)
    theta_0 = numpy.rad2deg(theta_0)
    norm = int(normalized)
    uf = numpy.frompyfunc(lambda x:mfunnorm.mfunnorm(m,tau,x,theta_0, norm)[1], 1, 1)

    if mfactor:
        return (-1)**m*numpy.float64(uf(theta))
    else:
        return numpy.float64(uf(theta))

def dmehler_t_numeric(m, tau, theta, theta_0=numpy.pi, delta=1e-8, normalized=False):

    theta = numpy.rad2deg(theta)
    theta_0 = numpy.rad2deg(theta_0)

    delta_a = theta_0*delta
    norm=int(normalized)

    uf = numpy.frompyfunc(lambda x:mfunnorm.mfunnorm(m,tau,x,theta_0, norm)[0], 1, 1)
    
    derivative = (numpy.float64(uf(theta + delta_a)) - numpy.float64(uf(theta - delta_a)))/(2*delta_a)
    return numpy.rad2deg(derivative)
    
    
