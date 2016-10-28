import numpy
import numpy.linalg
import scipy
import scipy.interpolate
import scipy.special
import scipy.optimize
import bspline
import xyzfield as xyz
import warnings
from matplotlib import pyplot
import sys

#debug
import memory_profiler

sin = numpy.sin; cos = numpy.cos; lpmv = scipy.special.lpmv


def dlpmv(m, n, z):
    
    """evaluate d/dz of the associated legendre function of order m and degree n at z """
    return 1/(z**2-1)*(n*z*lpmv(m,n,z) - (m+n)*lpmv(m, n-1, z))

def step_solve(func, interval, args=(), step=0.02) -> numpy.ndarray:

    """find all roots of func within interval using Brent's method, searching for zeros in increments of step"""
    roots = []

    for a, b in zip(numpy.arange(*interval, step), numpy.arange(*interval, step) + step):
        #check for sign change
        if (func(a, *args) * func(b, *args)) < 0.0:
            #solve within (a, b)
            roots.append(scipy.optimize.brentq(func, a, b, args))
            
    return numpy.array(roots)

def degree(ms, theta0, max_k, solve_step=0.01, overstep_size = 1.0) -> list:

    """find n[k], degrees of the legendre functions such that boundary conditions on theta0 are satisfied"""
    lpmv = scipy.special.lpmv
    cos = scipy.cos

    maxrange = max_k*10

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

def join_roots(roots):

    """merge the proper (k >= m) roots of the legendre functions as given by degree() into something usable in order\
to avoid death"""

    k_max = len(roots)

    k, m, n = [], [], []

    for ki in range(0, k_max):
        k.append(ki)
        m.append(0)
        for mi in range(1, ki + 1):
            k.append(ki)
            k.append(ki)
            m.append(mi)
            m.append(-mi)

    roots_joined = [numpy.sort(numpy.concatenate((evens, odds))) for (evens, odds) in roots]
    for mi, m_roots in enumerate(roots_joined):
        roots_joined[mi] = m_roots[m_roots > mi]

    for (ki, mi) in zip(k, m):
        n.append(roots_joined[abs(mi)][ki-abs(mi)])

    return k, m, n

def schmidt_real(ms, ns, grid=True) -> numpy.array:

    numpy.seterr(divide="ignore", invalid="ignore")
    if grid:
        n, m = numpy.meshgrid(ns, ms, indexing="ij")
        m = abs(m)
    else:
        n, m = numpy.atleast_1d(ns), abs(numpy.atleast_1d(ms))
    #factor = (-1)**m*scipy.sqrt((2-1.0*(m == 0))*scipy.special.gamma(n-m+1)/scipy.special.gamma(n+m+1))
    #factor = scipy.sqrt((2-1.0*(m == 0))*scipy.special.gamma(n-m+1)/scipy.special.gamma(n+m+1))
    factor = (-1)**m*scipy.sqrt((2-1.0*(m == 0))*scipy.special.gamma(n-m+1)/scipy.special.gamma(n+m+1))
    return numpy.squeeze(numpy.real(factor))

def rotation_matrix(theta_pole, phi_pole, invert = False) -> numpy.array:
    
    ry = numpy.array(((cos(theta_pole), 0. , sin(theta_pole)),
                      (0. , 1. , 0. ),
                      (-sin(theta_pole), 0. , cos(theta_pole))))

    #quitar la T
    rz = numpy.array(((cos(phi_pole), sin(phi_pole), 0.),
                     (-sin(phi_pole), cos(phi_pole), 0.),
                     (0. , 0. , 1. ))).T

    if invert:
        return numpy.linalg.inv(ry) @ numpy.linalg.inv(rz)
    else:
        return ry @ rz
    
def rotate_coords(r, theta, phi, matrix):

    #cartesian
    original = r*numpy.array((sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)))

    x_r, y_r, z_r = matrix @ original
    
    #convert to spherical
    theta_r = numpy.arccos(z_r/r)
    phi_r = numpy.arctan2(y_r, x_r)

    return (r, theta_r, phi_r)
    
def rotate_vector(x, y, z, pole_theta, pole_phi, theta, phi, theta_rot, invert = False):

    #guay
    angle = numpy.arcsin(sin(pole_theta)*sin(numpy.arctan2(sin(phi-pole_phi), cos(phi-pole_phi)))/sin(theta_rot))
    #if (pole_theta > theta):
    if(cos(theta_rot)*cos(theta) > cos(pole_theta)):
        angle = -angle + numpy.pi

    if invert: angle = -angle
    
    rot_mat = numpy.array(((cos(angle), -sin(angle), 0.),
                           (sin(angle), cos(angle), 0.),
                           (0. , 0. , 1.)))

    return rot_mat @ numpy.array((x,y,z))

def rotate_declination(dec, pole_theta, pole_phi, theta, phi, theta_rot, invert = False):

    #ok it works
    angle = numpy.arcsin(sin(pole_theta)*sin(numpy.arctan2(sin(phi-pole_phi), cos(phi-pole_phi)))/sin(theta_rot))
    #angle[pole_theta > theta] = -angle[pole_theta > theta] + numpy.pi
    angle[cos(theta_rot)*cos(theta) > cos(pole_theta)] = -angle[cos(theta_rot)*cos(theta) > cos(pole_theta)] \
                                                         + numpy.pi

    if invert: angle = -angle

    return dec + angle

def xyzfield(k, m, n, gcoefs, thetav, phiv):    
    lpmv = scipy.special.lpmv


    x = numpy.zeros_like(thetav)
    y = x.copy()
    z = x.copy()

    m_abs = numpy.abs(m)
    
    schmidt = schmidt_real(m_abs, n, grid=False)

    for ki, mi, ni, g, sch in zip(k, m, n, gcoefs, schmidt):
    
        m_abs = abs(mi)
    
        cossin = numpy.cos(m_abs*phiv) if mi >= 0 else numpy.sin(m_abs*phiv)
        sinmcos = numpy.sin(m_abs*phiv) if mi >= 0 else -numpy.cos(m_abs*phiv)

        leg = lpmv(m_abs, ni, numpy.cos(thetav))*sch
        dleg = dlpmv(m_abs, ni, numpy.cos(thetav))*(-numpy.sin(thetav))*sch
    
        #x += g*cossin*sch*dlpmv(m_abs, ni, numpy.cos(thetav))*(-numpy.sin(thetav))
        #y += g*sinmcos*m_abs*sch*lpmv(m_abs, ni, numpy.cos(thetav))/numpy.sin(thetav)
        #z -= (ki+1)*(g*cossin)*sch*lpmv(m_abs, ni, numpy.cos(thetav))

        x += g*cossin*dleg
        y += g*sinmcos*m_abs*leg/numpy.sin(thetav)
        z -= (ni+1)*(g*cossin)*leg

    return x, y, z

def polar_contour(scalar, thetav, phiv, theta0, ax, resolution=200, base=None):
    
    lat0 = numpy.rad2deg(theta0)
    from mpl_toolkits.basemap import Basemap
    
    if base is None:
        base = Basemap(projection="npaeqd", lon_0 = 0, boundinglat=90-lat0)

    #base.drawparallels(numpy.arange(90-lat0, 90, 5), ax=ax)

    phinew=scipy.linspace(-numpy.pi, numpy.pi, resolution)
    thetanew=scipy.linspace(0.01*theta0, theta0, resolution)

    thetagrid, phigrid = scipy.meshgrid(thetanew,phinew,indexing="xy")

    xx=scipy.interpolate.griddata((thetav,phiv), scalar, (thetagrid,phigrid), method="linear")
    xx[numpy.isnan(xx)] = 0.0

    lon = numpy.rad2deg(phigrid)
    lat = 90 - numpy.rad2deg(thetagrid)

    return base.contourf(lon, lat, xx, 40, latlon=True, ax=ax, cmap="bwr")

def polar_tricontour(scalar, thetav, phiv, theta0, ax, base=None, cmap="bwr", scale="symmetric"):
    
    lat0 = numpy.rad2deg(theta0)
    from mpl_toolkits.basemap import Basemap
    
    if base is None:
        base = Basemap(projection="npaeqd", lon_0 = 0, boundinglat=90-lat0)

    #base.drawparallels(numpy.arange(90-lat0, 90, 5), ax=ax)

    lonv = numpy.rad2deg(phiv)
    latv = 90 - numpy.rad2deg(thetav)
    
    x_coord, y_coord = base(lonv, latv)

    if(scale == "symmetric"):
        vmin=-numpy.max(abs(scalar))*1.1; vmax=numpy.max(abs(scalar))*1.1
    elif(scale == "minmax"):
        vmin=numpy.min(scalar); vmax=numpy.max(scalar)
    elif(scale == "positive"):
        vrange = numpy.max(scalar)-numpy.min(scalar)
        vmax = numpy.max(scalar); vmin = vmax-vrange*2
    else:
        vmin=-numpy.max(abs(scalar))*1.1; vmax=numpy.max(abs(scalar))*1.1

    return ax.tricontourf(x_coord, y_coord, scalar, 60, cmap=cmap,
                          vmin=vmin, vmax=vmax)

def condition_matrix_xyz(thetav, phiv, degrees):

    k, m, n = numpy.array(degrees)
    m_abs = numpy.abs(m)
    cos = numpy.cos; sin = numpy.sin; lpmv = scipy.special.lpmv
    schmidt = schmidt_real(m_abs, n, grid=False)

    cossin = numpy.zeros((len(thetav), len(k)))
    cossin[:, m >= 0] = cos(phiv[:, numpy.newaxis] @ numpy.abs(m[numpy.newaxis, :]))[:, m >= 0]
    cossin[:, m < 0] = sin(phiv[:, numpy.newaxis] @ numpy.abs(m[numpy.newaxis, :]))[:, m < 0]

    sinmcos = numpy.zeros_like(cossin)
    sinmcos[:, m >= 0] = sin(phiv[:, numpy.newaxis] @ numpy.abs(m[numpy.newaxis, :]))[:, m >= 0]
    sinmcos[:, m < 0] = -cos(phiv[:, numpy.newaxis] @ numpy.abs(m[numpy.newaxis, :]))[:, m < 0]

    costhetav = cos(thetav)[:, numpy.newaxis]
    sinthetav = sin(thetav)[:, numpy.newaxis]

    leg = numpy.zeros_like(cossin)
    dleg = numpy.zeros_like(cossin)

    for i, (mi_abs, ni, sch) in enumerate(zip(m_abs,n,schmidt)):
        leg[:, i] = lpmv(mi_abs, ni, costhetav[:,0])*sch
        #dleg[:, i] = dlpmv(mi, ni, costhetav[:,0])*sch
        #dleg[:, i]  = numpy.gradient(leg[:,i])*len(thetav)/numpy.max(thetav)
        dleg[:, i] = dlpmv(mi_abs, ni, costhetav[:,0])*sch*(-sinthetav[:,0])
        
    Ax = cossin * dleg
    Ay = m_abs * leg * sinmcos / sinthetav
    Az = -(n + 1) * leg * cossin

    return Ax, Ay, Az

def invert_xyz(thetav, phiv, Bx, By, Bz, degrees):

    #build condition matrix
    Axyz = numpy.concatenate(condition_matrix_xyz(thetav, phiv, degrees), axis=0)

    Bxyz = numpy.concatenate((Bx, By, Bz), axis=0)

    #do a nice invert
    g = (numpy.linalg.inv(Axyz.T @ Axyz) @ Axyz.T) @ Bxyz
    #g = numpy.linalg.lstsq(Axyz, Bxyz)[0]

    return g

def condition_matrix_dif(x, y, z, f, h, Ax, Ay, Az):

    xx = x[:, numpy.newaxis]; yy = y[:, numpy.newaxis]; zz = z[:, numpy.newaxis]
    ff = f[:, numpy.newaxis]; hh = h[:, numpy.newaxis]

    Ad = (-yy*Ax + xx*Ay)/hh**2
    Ai = (-xx*zz*Ax - yy*zz*Ay)/(hh*ff**2)+Az*hh/ff**2
    Af = (xx*Ax+yy*Ay+zz*Az)/ff
    
    return Ad, Ai, Af

def invert_dif(thetav, phiv, D, I, F, degrees, g0=None, steps=5):

    # reescalar intensidad pls

    k, m, n = degrees
    
    if g0 is None:
        g0 = numpy.zeros_like(k); g0[0] = -1000 #because why the heck not

    g = g0.copy()

    Ax, Ay, Az = condition_matrix_xyz(thetav, phiv, degrees)

    D_0, I_0, F_0 = D, I, F

    D_0 = D_0[~numpy.isnan(D_0)]
    I_0 = I_0[~numpy.isnan(I_0)]
    F_0 = F_0[~numpy.isnan(F_0)]
    
    #dif_0 = numpy.concatenate((D_0, I_0, F_0))
    di_0 = numpy.concatenate((D_0, I_0))

    for iteration in range(steps):
        Bx, By, Bz = xyzfield(k, m, n, g, thetav, phiv)

        D_model, I_model, F_model, H_model = xyz.xyz2difh(Bx, By, Bz)

        Ad, Ai, Af = condition_matrix_dif(Bx, By, Bz, F_model, H_model, Ax, Ay, Az)
        Af = Af / F_model[:, numpy.newaxis]
        Adif = numpy.concatenate((Ad[~numpy.isnan(D_0), :], Ai[~numpy.isnan(I_0), :], Af[~numpy.isnan(F_0), :]),
                                 axis=0)

        D_model = D_model[~numpy.isnan(D_0)]
        I_model = I_model[~numpy.isnan(I_0)]
        F_model = F_model[~numpy.isnan(F_0)]
        
        di = numpy.concatenate((D_model, I_model), axis=0)

        #corregir diferencias 360 -> 0
        di_delta = numpy.arctan2(numpy.sin(di-di_0), numpy.cos(di-di_0))
        f_delta = (F_model - F_0)/F_model

        delta = numpy.concatenate((di_delta, f_delta), axis=0)

        #hmmmmm
        g = g - (numpy.linalg.pinv(Adif.T @ Adif) @ Adif.T) @ delta
        #solution = numpy.linalg.lstsq(Adif, delta)
        #g = g - solution[0]
        sys.stdout.write("\r")
        sys.stdout.write(str(numpy.sqrt(sum(delta**2))))
        #DEBUGE
        
        #print(g[:4])

    if (sum(abs(delta**2)) > 10000): warnings.warn("inversion might not have converged (╯°□°)╯︵ ┻━┻")
    return g

def xyzfieldt(k, m, n, knots, gcoefs, thetav, phiv, t):

    lpmv = scipy.special.lpmv

    x = numpy.zeros_like(thetav)
    y = x.copy()
    z = x.copy()

    schmidt = schmidt_real(m, n, grid=False)
    
    #1. calcular los splines
    #base = bspline.deboor_base(knots, t, 3).T[:-4]
    base = bspline.condition_array(knots, t).T
    #3. for q...
    for q, spline in enumerate(base):
        for ki, mi, ni, g, sch in zip(k, m, n, gcoefs[q,:], schmidt):
    
            m_abs = abs(mi)
    
            cossin = numpy.cos(m_abs*phiv) if mi >= 0 else numpy.sin(m_abs*phiv)
            sinmcos = numpy.sin(m_abs*phiv) if mi >= 0 else -numpy.cos(m_abs*phiv)

            leg = lpmv(m_abs, ni, numpy.cos(thetav))*sch
            dleg = dlpmv(m_abs, ni, numpy.cos(thetav))*(-numpy.sin(thetav))*sch
    
            x += g*cossin*dleg*spline
            y += g*sinmcos*m_abs*leg/numpy.sin(thetav)*spline
            z -= (ni+1)*(g*cossin)*leg*spline

    return x, y, z

    
#@memory_profiler.profile
def invert_dift(thetav, phiv, t, D, I, F, degrees, knots, g0=None, steps=5):

    k, m, n = degrees

    #n_knots, n_degrees = len(knots)-4, len(k)
    #knots = bspline.fix_knots(kn, 3)
    
    n_knots, n_degrees = len(knots), len(k)
    n_coefs = n_knots*n_degrees
    #base = bspline.deboor_base(knots, t, 3)[:, :-4]

    if g0 is None:
        g0 = numpy.zeros((n_knots, n_degrees))
        g0[:, 0] = -1000.0

    g = g0.copy()

    Ax, Ay, Az = condition_matrix_xyz(thetav, phiv, degrees)
    D_0, I_0, F_0 = D, I, F

    base = numpy.concatenate((bspline.condition_array(knots, t[~numpy.isnan(D)]),
                              bspline.condition_array(knots, t[~numpy.isnan(I)]),
                              bspline.condition_array(knots, t[~numpy.isnan(F)])), axis=0)

    D_0 = D_0[~numpy.isnan(D)]
    I_0 = I_0[~numpy.isnan(I)]
    F_0 = F_0[~numpy.isnan(F)]
    
    di_0 = numpy.concatenate((D_0, I_0))

    for iteration in range(steps):
        
        #como poner la dependencia del tiempo aquí?
        #con magia
        Bx, By, Bz = xyzfieldt(k, m, n, knots, g, thetav, phiv, t)
        D_model, I_model, F_model, H_model = xyz.xyz2difh(Bx, By, Bz)

        Ad, Ai, Af = condition_matrix_dif(Bx, By, Bz, F_model, H_model, Ax, Ay, Az)
        Af = Af / F_model[:, numpy.newaxis]
        Adif = numpy.concatenate((Ad[~numpy.isnan(D), :],
                                  Ai[~numpy.isnan(I), :],
                                  Af[~numpy.isnan(F), :]), axis=0)

        #quick hacc
        Adif = numpy.concatenate([Adif*spline[:, numpy.newaxis] for spline in base.T], axis=1)
                           
        D_model = D_model[~numpy.isnan(D)]
        I_model = I_model[~numpy.isnan(I)]
        F_model = F_model[~numpy.isnan(F)]

        di = numpy.concatenate((D_model, I_model), axis=0)
        di_delta = numpy.arctan2(numpy.sin(di-di_0), numpy.cos(di-di_0))
        f_delta = (F_model - F_0)/F_model

        delta = numpy.concatenate((di_delta, f_delta), axis=0)

        solution = numpy.linalg.lstsq(Adif, delta)
        g = g - solution[0].reshape((n_knots, n_degrees))
        #g = g - ((numpy.linalg.pinv(Adif.T @ Adif) @ Adif.T) @ delta).reshape((n_knots, n_degrees))

        sys.stdout.write("\r")
        sys.stdout.write("iteration {0}  : rms = ".format(iteration+1))
        sys.stdout.write(str(numpy.sqrt(sum(delta**2)/len(delta))))

        #debuge

        #ax.scatter(t, di_delta[:len(di_delta)//2], color="green")
        #ax.scatter(t, di_delta[len(di_delta)//2:], color="red")
        #ax.scatter(t, f_delta, color="blue")

        #ax.scatter(t, D_model - D_0, color="green")
        #ax.scatter(t, I_model - I_0, color="red")
        #ax.scatter(t, F_model - F_0, color="blue")

        #pyplot.show(fig)
        
    if (sum(abs(delta**2)) > 10000): warnings.warn("a bad thing is happening ヽ( `д´*)ノ")
    return g
