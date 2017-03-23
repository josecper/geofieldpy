import numpy

def curve_at(times, r, theta, phi):

    return times, numpy.ones_like(times)*r, numpy.ones_like(times)*theta, numpy.ones_like(times)*phi

def anything(rv, thetav, phiv, times):

    nd = len(thetav)
    ntimes = len(times)

    theta_d = numpy.tile(thetav, ntimes)
    phi_d = numpy.tile(phiv, ntimes)
    r_d = numpy.tile(rv, ntimes)
    times_d = numpy.repeat(times, nd)

    return times_d, r_d, theta_d, phi_d
