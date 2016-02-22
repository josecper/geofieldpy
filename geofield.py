import numpy
import scipy

def interpolatecoefs(gcoefsd,hcoefsd,times):
    """
    gcoefsd = dictionary of { float : m*l float64 array }
    hcoefsd = same
    times = array of years

    returns g,h : m*l*t.shape
    """

    
