import numpy
import numpy.linalg
import scipy
import scipy.special
import scipy.optimize
import xyzfield

sin = numpy.sin; cos = numpy.cos

def step_solve(func, interval, args=(), step=0.02) -> numpy.ndarray:

    """find all roots of func within interval using Brent's method, searching for zeros in increments of step"""
    roots = []

    for a, b in zip(numpy.arange(*interval, step), numpy.arange(*interval, step) + step):
        #check for sign change
        if (func(a, *args) * func(b, *args)) < 0.0:
            #solve within (a, b)
            roots.append(scipy.optimize.brentq(func, a, b, args))
            
    return numpy.array(roots)

def degree(ms, theta0, max_k, solve_step=0.02, overstep_size = 1.0) -> list:

    """find n[k], degrees of the legendre functions such that boundary conditions on theta0 are satisfied"""
    lpmv = scipy.special.lpmv
    cos = scipy.cos

    #arbitrary magical number
    maxrange = max_k*3

    #bendito sea el señor
    ms = numpy.atleast_1d(ms)
    roots = []

    for m in ms:
        def polyval(n):
            #value of the legendre function
            return lpmv(m, n, cos(theta0))

        def polyder(n):
            #value of the legendre function derivative
            z = cos(theta0)
            return 1/(z**2-1)*(z*n*lpmv(m,n,z) - (m+n)*lpmv(m,n-1,z))

        odd_roots, even_roots = (step_solve(polyval, (-0.1, maxrange), step=solve_step),
                                 step_solve(polyder, (-0.1, maxrange), step=solve_step))

        #check for still unfound roots egg?
        overstep = overstep_size
        while len(odd_roots)+len(even_roots) < max_k:
            numpy.append(odd_roots, step_solve(polyval, (maxrange - 0.1, maxrange + overstep), step=solve_step))
            numpy.append(even_roots, step_solve(polyder, (maxrange - 0.1, maxrange + overstep), step=solve_step))
            overstep += overstep_size
            
        roots.append(numpy.array((odd_roots, even_roots)))

    return roots

def join_roots(ms, roots):
    #create degree array
    pass

def schmidt_real(ms, ns) -> numpy.array:

    numpy.seterr(divide="ignore", invalid="ignore")
    n, m = numpy.meshgrid(ns, ms, indexing="ij")
    factor = (-1)**m*scipy.sqrt((2-1.0*(m == 0))*scipy.special.gamma(n-m+1)/scipy.special.gamma(n+m+1))
    return numpy.squeeze(numpy.real(factor))

def rotation_matrix(theta_pole, phi_pole, invert = False) -> numpy.array:
    
    ry = numpy.array(((cos(theta_pole), 0. , sin(theta_pole)),
                      (0. , 1. , 0. ),
                      (-sin(theta_pole), 0. , cos(theta_pole))))

    #quitar la T por la gloria de tu madrer
    rz = numpy.array(((cos(phi_pole), sin(phi_pole), 0. ),
                     (-sin(phi_pole), cos(phi_pole), 0. ),
                     (0. , 0. , 1. ))).T

    if invert:
        return numpy.linalg.inv(ry) @ numpy.linalg.inv(rz)
    else:
        return ry @ rz
    
def rotate_coords(r, theta, phi, matrix):

    #cartesian
    original = r*numpy.array((sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)))

    #do a magic
    x_r, y_r, z_r = matrix @ original
    
    #convert to spherical
    theta_r = numpy.arccos(z_r/r)
    phi_r = numpy.arctan2(y_r, x_r)

    return (r, theta_r, phi_r)
    
def rotate_vector(x, y, z, pole_theta, pole_phi, theta, phi, theta_rot):

    #NO SE SI ESTÁ BIEN PERO PODRÍA SER QUE SÍ
    #SÍ ESTÁ BIEN AL 96.4% DE CONFIANZA
    #this will be slow since the rotation matrix depends on each point's coordinates
    #and so it can't be precomputed :'(
    angle = numpy.arcsin(sin(pole_theta)*sin(numpy.arctan2(sin(phi-pole_phi), cos(phi-pole_phi)))/sin(theta_rot))
    if (pole_theta > theta):
        angle = -angle + numpy.pi
    
    rot_mat = numpy.array(((cos(angle), -sin(angle), 0.),
                           (sin(angle), cos(angle), 0.),
                           (0. , 0. , 1.)))

    return rot_mat @ numpy.array((x,y,z))
