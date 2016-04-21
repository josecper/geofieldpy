import numpy
import scipy, scipy.constants

gr = scipy.constants.golden_ratio
ga = 2*numpy.pi*gr

def grid(n=10000, sort=True):
    """
    returns thetav, phiv spherical angles on an approximately regular fibonacci sphere.
    options:
    · n : number of points
    · sort : sorts points by latitude. useful so basemap doesn't crash.
    """
    index = numpy.arange(1,n)
    phiv = (ga*index) % (2*numpy.pi)
    phiv -= 2*numpy.pi*(phiv > numpy.pi)
    
    thetav = numpy.pi/2-scipy.arcsin(-1+2*index/n)
    if sort:
        sortindex = numpy.argsort(phiv)
        return thetav[sortindex], phiv[sortindex]
    else:
        return thetav, phiv
