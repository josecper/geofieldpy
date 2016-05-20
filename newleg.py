from itertools import chain
import scipy, scipy.misc
from scipy.special import lpmn
import numpy

def degrees(order):

    l=numpy.array(list(chain(*[[i]*(2*i+1) for i in range(order+1)])))
    m=numpy.array(list(chain(*chain(*[[[i, -i] if i != 0 else [0] for i in range(j+1)] for j in range(order+1)]))))

    return m,l
    
def schmidt(m,l):

    return (-1)**m*scipy.sqrt((2-(m == 0))*scipy.misc.factorial(l-m)/scipy.misc.factorial(l+m))

def legendre(xv, order):

    legout=numpy.zeros((order*(order+2)+1,len(xv)))
    dlegout=legout.copy()
    m,l=degrees(order)
    sch=schmidt(m,l)
    
    for i,x in enumerate(xv):
        leg, dleg = lpmn(order,order,x)
        legout[:,i]= leg[m,l]*sch
        dlegout[:,i]= dleg[m,l]*sch

    return legout, dlegout
