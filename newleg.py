from itertools import chain
from joblib import Memory
import scipy, scipy.misc
from scipy.special import lpmn
import numpy

mem = Memory(cachedir="/tmp/joblib", verbose=0)

@mem.cache
def degrees(order, start=0):

    l=numpy.array(list(chain(*[[i]*(2*i+1) for i in range(start,order+1)])))
    m=numpy.array(list(chain(*chain(*[[[i, -i] if i != 0 else [0] for i in range(j+1)] for j in range(start,order+1)]))))

    return m,l

@mem.cache
def schmidt(m,l):

    m_abs=numpy.abs(m)
    return (-1)**m_abs*scipy.sqrt((2-(m_abs == 0))*scipy.misc.factorial(l-m_abs)/scipy.misc.factorial(l+m_abs))

@mem.cache
def legendre(xv, order):

    legout=numpy.zeros((order*(order+2),len(xv)))
    dlegout=legout.copy()
    m,l=degrees(order, start=1)
    sch=schmidt(m,l)
    
    for i,x in enumerate(xv):
        leg, dleg = lpmn(order,order,x)
        legout[:,i]= leg[abs(m),l]*sch
        dlegout[:,i]= dleg[abs(m),l]*sch

    return legout, dlegout
