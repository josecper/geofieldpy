import numpy
import scipy, scipy.constants

gr = scipy.constants.golden_ratio
ga = 2*numpy.pi*gr

def grid(n, sort=True):

    index = numpy.arange(1,n)
    phiv = (ga*index) % (2*numpy.pi)
    phiv -= 2*numpy.pi*(phiv > numpy.pi)
    
    thetav = numpy.pi/2-scipy.arcsin(-1+2*index/n)
    if sort:
        sortindex = numpy.argsort(phiv)
        return thetav[sortindex], phiv[sortindex]
    else:
        return thetav, phiv
