import numpy, scipy
import mehler
import scha
import xyzfield
import sys

from numpy import cos, sin
from scipy.special import lpmv
from scha import condition_matrix_dif

a_r = 6371.2
a_ellip = 6378.137
b_ellip = 6356.752

def rscha_condition_matrix_dif(kmn_leg_in, kmn_leg_ex, m_mehler, rv, thetav, phiv, theta_0, Bx_ref, By_ref, Bz_ref, normalize=True):

    k_in, m_in, n_in = kmn_leg_in
    k_ex, m_ex, n_ex = kmn_leg_ex
    ncoefs_in = len(k_in)
    ncoefs_ex = len(k_ex)
    ncoefs_meh = len(m_mehler)
    ndatos = len(rv)
    
    kmn_empty = ((), (), ())
    Ax_leg_in = numpy.zeros((ndatos, ncoefs_in))
    Ay_leg_in = Ax_leg_in.copy()
    Az_leg_in = Ax_leg_in.copy()

    for i, (ki, mi, ni) in enumerate(zip(k_in, m_in, n_in)):
        kmni = ((ki,), (mi,), (ni,))
        Ax_leg_in[:, i], Ay_leg_in[:, i], Az_leg_in[:, i] = leg_field(kmni, kmn_empty,
                                                                      (1,), (), rv, thetav, phiv)

    Ax_leg_ex = numpy.zeros((ndatos, ncoefs_ex))
    Ay_leg_ex = Ax_leg_ex.copy()
    Az_leg_ex = Ax_leg_ex.copy()

    for i, (ki, mi, ni) in enumerate(zip(k_ex, m_ex, n_ex)):
        kmni = ((ki,), (mi,), (ni,))
        Ax_leg_ex[:, i], Ay_leg_ex[:, i], Az_leg_ex[:, i] = leg_field(kmn_empty, kmni,
                                                                      (), (1,), rv, thetav, phiv)

    Ax_meh = numpy.zeros((ndatos, ncoefs_meh))
    Ay_meh = Ax_meh.copy()
    Az_meh = Ax_meh.copy()

    for i, mi in enumerate(m_mehler):
        Ax_meh[:, i], Ay_meh[:, i], Az_meh[:, i] = mehler_field((mi,), (1,), rv, thetav, phiv, theta_0)
    
    Ax = numpy.concatenate((Ax_leg_in, Ax_leg_ex, Ax_meh), axis=1)
    Ay = numpy.concatenate((Ay_leg_in, Ay_leg_ex, Ay_meh), axis=1)
    Az = numpy.concatenate((Az_leg_in, Az_leg_ex, Az_meh), axis=1)
    #Axyz = numpy.concatenate((Ax, Ay, Az), axis=0)

    del Ax_meh, Ay_meh, Az_meh, Ax_leg_ex, Ay_leg_ex, Az_leg_ex, Ax_leg_in, Ay_leg_in, Az_leg_in

    # ahora a hacer el dif
    # normalizar el campo
    F_ref_n = numpy.sqrt(Bx_ref**2 + By_ref**2 + Bz_ref**2)

    if normalize:
        F_avg = numpy.average(F_ref_n)
    else:
        F_avg = 1
        
    F_ref_n = F_ref_n / F_avg
    Bx_ref_n = Bx_ref / F_avg
    By_ref_n = By_ref / F_avg
    Bz_ref_n = Bz_ref / F_avg
    H_ref_n = numpy.sqrt(Bx_ref_n**2+By_ref_n**2)

    Adif = scha.condition_matrix_dif(Bx_ref_n, By_ref_n, Bz_ref_n, F_ref_n, H_ref_n, Ax, Ay, Az)
    return Adif
    

def mehler_condition_matrix_xyz(rv, thetav, phiv, m, theta_0):
    """
    Mehler functions base condition matrix [(3*ndata) x ncoefs]. This is to be
    concatenated to the usual Legendre condition matrix to solve the inverse
    problem.
    """

    # esto es la parte que va multiplicada por G, H.
    
    #X = 1/r*dV/dtheta
    Ax = numpy.zeros ((len(thetav), len(m)))
    #Y = -1/rsintheta * dV/dphi
    Ay = Ax.copy()
    #Z = 0 en el caso de r=a (ver notas)
    Az = Ax.copy()

    for i, mm in enumerate(m):
        mm_abs = numpy.abs(mm)
        #mehler_dp = mehler.dmehler_t(mm_abs, 0, thetav, theta_0, normalized=False) #MAGIC
        mehler_dp = mehler.dmehler_t(mm_abs, 0, thetav, theta_0, normalized=True, mfactor=True)
        mehler_p = mehler.mehler_t(mm_abs, 0, thetav, theta_0, normalized=True, mfactor=True)

        cossin = cos(mm_abs*phiv) if mm >= 0 else sin(mm_abs*phiv)
        sinmcos = sin(mm_abs*phiv) if mm >= 0 else -cos(mm_abs*phiv)
        
        Ax[:,i] = r_mehler(rv)*mehler_dp*cossin
        Ay[:,i] = r_mehler(rv)*mm_abs/sin(thetav)*mehler_p*sinmcos
        Az[:,i] = dr_mehler(rv)*mehler_p*cossin #magia chunga

    Axyz = numpy.concatenate ((Ax,Ay,Az), axis=0)
    return Axyz

def legendre_condition_matrix_xyz(rv, thetav, phiv, degrees):

    #wRONG

    k, m, n = numpy.array(degrees)
    m_abs = numpy.abs(m)
    cos = numpy.cos; sin = numpy.sin; lpmv = scipy.special.lpmv
    schmidt = numpy.atleast_1d(scha.schmidt_real(m_abs, n, grid=False))

    cossin = numpy.zeros((len(thetav), len(k)))
    cossin[:, m >= 0] = cos(phiv[:, numpy.newaxis] @ numpy.abs(m[numpy.newaxis, :]))[:, m >= 0]
    cossin[:, m < 0] = sin(phiv[:, numpy.newaxis] @ numpy.abs(m[numpy.newaxis, :]))[:, m < 0]

    sinmcos = numpy.zeros_like(cossin)
    sinmcos[:, m >= 0] = sin(phiv[:, numpy.newaxis] @ numpy.abs(m[numpy.newaxis, :]))[:, m >= 0]
    sinmcos[:, m < 0] = -cos(phiv[:, numpy.newaxis] @ numpy.abs(m[numpy.newaxis, :]))[:, m < 0]

    costhetav = cos(thetav)[:, numpy.newaxis]
    sinthetav = sin(thetav)[:, numpy.newaxis]
    rsa = (a_r/rv)[:, numpy.newaxis]

    leg = numpy.zeros_like(cossin)
    dleg = numpy.zeros_like(cossin)
    ra = leg.copy()
    dra = leg.copy()
    
    for i, (mi_abs, ni, sch) in enumerate(zip(m_abs,n,schmidt)):
        leg[:, i] = lpmv(mi_abs, ni, costhetav[:,0])*sch
        dleg[:, i] = scha.dlpmv(mi_abs, ni, costhetav[:,0])*sch*(-sinthetav[:,0])
        ra[:, i] = rsa[:,0]**(ni+1)
        dra[:,i] = rsa[:,0]**(ni+2)
        
    Ax = cossin * dleg * ra
    Ay = m_abs * leg * sinmcos / sinthetav * ra
    Az = -(n + 1) * leg * cossin * dra

    return Ax, Ay, Az

#ha ha whoops you died oh well

def norm_leg_even(k, m, n, theta_0):

    m_abs = numpy.abs(m)
    schmidt = scha.schmidt_real(m_abs, n, grid=False)

    return (-sin(theta_0))**2*lpmv(m_abs, n, cos(theta_0))*scha.dndlpmv(m_abs, n, cos(theta_0))*schmidt**2\
        /(2*n+1)

def rscha_spatial_reg_diag(kmn_in, kmn_ext, m_mehler, theta_0):
    """
    Legendre/Mehler functions spatial regularization matrix (diagonal?? hmm), 
    based on the <B²> norm.
    """
    k_even = numpy.concatenate((kmn_in[0], kmn_ext[0]))
    m_even = numpy.concatenate((kmn_in[1], kmn_ext[1]))
    n_even = numpy.concatenate((kmn_in[2], kmn_ext[2]))
    
    n_leg = len(k_even)
    n_mehler = len(m_mehler)

    n_coefs = n_leg + n_mehler

    mp = (m_even >= 0)
    mn = (m_even < 0)

    c1 = numpy.zeros(n_coefs)
    c2 = c1.copy()
    c3 = c1.copy()
    leg_norm = norm_leg_even(k_even, m_even, n_even, theta_0)

    c1[:n_leg][mp] = (2*n_even[mp]**2+3*n_even[mp]+1)*(1+(m_even[mp] == 0))*numpy.pi*leg_norm[mp]
    c1[:n_leg][mn] = (2*n_even[mn]**2+n_even[mn])*numpy.pi*leg_norm[mn]

    for i, mm in enumerate(m_mehler):
        mm_abs = numpy.abs(mm)
        c2[n_leg+i] = (mehler.mehler_t(mm_abs, 0, theta_0, theta_0, normalized=True)*\
                       mehler.dmehler_t(mm_abs, 0, theta_0, theta_0, normalized=True)*numpy.sin(theta_0)-\
                       0.25)*(1+(mm == 0))*numpy.pi

    return numpy.diag(c1+c2)
    
def rscha_time_reg_diag(Mreg_spatial, knots):
    """
    Mehler functions temporal reg. matrix, just multiply copies of thing by a
    diff diff eye.
    """
    n_knots = len(knots)
    n_degrees = Mreg_spatial.shape[0]
    
    d2 = numpy.diff(numpy.eye(n_knots),2,axis=0)
    d2 = d2.T @ d2

    reg = numpy.tile(Mreg_spatial, (n_knots, n_knots))*\
          numpy.repeat(numpy.repeat(d2, n_degrees, axis=0), n_degrees, axis=1)
    
    return reg

def invert_xyz(thetav, phiv, Bx, By, Bz, kmn_legendre, m_mehler, theta_0):

    #matrices de condicion
    Axyz_legendre = numpy.concatenate(scha.condition_matrix_xyz(thetav, phiv, kmn_legendre), axis=0)
    Axyz_mehler = mehler_condition_matrix_xyz(thetav, phiv, m_mehler, theta_0)

    Axyz = numpy.concatenate((Axyz_legendre, Axyz_mehler), axis=1)
    Bxyz = numpy.concatenate((Bx, By, Bz), axis=0)
    del Axyz_legendre, Axyz_mehler

    g = numpy.linalg.lstsq(Axyz, Bxyz)[0]

    return g

def invert_xyz_reg(thetav, phiv, Bx, By, Bz, kmn_legendre, m_mehler, theta_0, reg_factor):

    Axyz_legendre = numpy.concatenate(scha.condition_matrix_xyz(thetav, phiv, kmn_legendre), axis=0)
    Axyz_mehler = mehler_condition_matrix_xyz(thetav, phiv, m_mehler, theta_0)

    Axyz = numpy.concatenate((Axyz_legendre, Axyz_mehler), axis=1)
    Bxyz = numpy.concatenate((Bx, By, Bz), axis=0)
    del Axyz_legendre, Axyz_mehler

    Mreg = rscha_spatial_reg_diag(kmn_legendre, m_mehler, theta_0)

    g = numpy.linalg.lstsq(Axyz.T @ Axyz + reg_factor*Mreg,
                           Axyz.T @ Bxyz)[0]

    return g

def invert_dif(thetav, phiv, D, I, F, kmn_legendre, m_mehler, theta_0, g0=-1000, steps=25):

    n_leg = len(kmn_legendre[0])
    n_mehler = len(m_mehler)
    n_coefs = n_leg + n_mehler
    n_data = len(thetav)
    
    g = numpy.zeros (n_coefs)
    g[0] = g0

    Axyz_legendre = numpy.concatenate(scha.condition_matrix_xyz(thetav, phiv, kmn_legendre), axis=0)
    Axyz_mehler = mehler_condition_matrix_xyz(thetav, phiv, m_mehler, theta_0)

    Axyz = numpy.concatenate((Axyz_legendre, Axyz_mehler), axis=1)
    Ax = Axyz[:n_data, :]; Ay = Axyz[n_data:2*n_data, :]; Az = Axyz[2*n_data:, :]
    di_0 = numpy.concatenate((D, I))

    for iteration in range(steps):
        Bx, By, Bz = synth_field (kmn_legendre, m_mehler, g, thetav, phiv, theta_0)
        D_model, I_model, F_model, H_model = xyzfield.xyz2difh(Bx, By, Bz)

        Ad, Ai, Af = scha.condition_matrix_dif(Bx, By, Bz, F_model, H_model, Ax, Ay, Az)
        Af = Af / F_model[:, numpy.newaxis]
        Adif = numpy.concatenate ((Ad, Ai, Af), axis=0)

        # voy por aqui
        di = numpy.concatenate((D_model, I_model), axis=0)
        di_delta = numpy.arctan2(numpy.sin(di-di_0), numpy.cos(di-di_0))
        f_delta = (F_model - F)/F_model
        delta = numpy.concatenate((di_delta, f_delta), axis=0)

        #newton-rahpson
        dif_g = numpy.linalg.lstsq(Adif, -delta)[0]
        #dif_g = numpy.linalg.lstsq(Adif.T @ Adif, -Adif.T @ delta)[0]
        g = g + dif_g
        
        sys.stdout.write("\r")
        sys.stdout.write(str(numpy.sqrt(sum(delta**2))))
        #yeah
            
    return g

def invert_dif_reg(thetav, phiv, D, I, F, kmn_legendre, m_mehler, theta_0, g0=-1000, steps=25, reg_coef=0.0):

    n_leg = len(kmn_legendre[0])
    n_mehler = len(m_mehler)
    n_coefs = n_leg + n_mehler
    n_data = len(thetav)
    
    g = numpy.zeros (n_coefs)
    g[0] = g0

    #regularización espacial del espacio
    reg_matrix = rscha_spatial_reg_diag(kmn_legendre, m_mehler, theta_0)*reg_coef

    Axyz_legendre = numpy.concatenate(scha.condition_matrix_xyz(thetav, phiv, kmn_legendre), axis=0)
    Axyz_mehler = mehler_condition_matrix_xyz(thetav, phiv, m_mehler, theta_0)

    Axyz = numpy.concatenate((Axyz_legendre, Axyz_mehler), axis=1)
    Ax = Axyz[:n_data, :]; Ay = Axyz[n_data:2*n_data, :]; Az = Axyz[2*n_data:, :]
    di_0 = numpy.concatenate((D, I))

    for iteration in range(steps):
        Bx, By, Bz = synth_field (kmn_legendre, m_mehler, g, thetav, phiv, theta_0)
        D_model, I_model, F_model, H_model = xyzfield.xyz2difh(Bx, By, Bz)

        Ad, Ai, Af = scha.condition_matrix_dif(Bx, By, Bz, F_model, H_model, Ax, Ay, Az)
        Af = Af / F_model[:, numpy.newaxis]
        Adif = numpy.concatenate ((Ad, Ai, Af), axis=0)

        # voy por aqui
        di = numpy.concatenate((D_model, I_model), axis=0)
        di_delta = numpy.arctan2(numpy.sin(di-di_0), numpy.cos(di-di_0))
        f_delta = (F_model - F)/F_model
        delta = numpy.concatenate((di_delta, f_delta), axis=0)

        #newton-rahpson
        dif_g = numpy.linalg.lstsq(Adif.T @ Adif + reg_matrix, -Adif.T @ delta + reg_matrix @ g)[0]
        g = g + dif_g
        
        sys.stdout.write("\r")
        sys.stdout.write(str(numpy.sqrt(sum(delta**2))))
        #yeah
            
    return g

def leg_field(kmn_in, kmn_ext, gcoefs_in, gcoefs_ext, rv, thetav, phiv):

    k_in, m_in, n_in = kmn_in
    k_ext, m_ext, n_ext = kmn_ext
    
    lpmv = scipy.special.lpmv

    x = numpy.zeros_like(thetav)
    y = x.copy()
    z = x.copy()
  
    schmidt_in = numpy.atleast_1d(scha.schmidt_real(m_in, n_in, grid=False))
    schmidt_ext = numpy.atleast_1d(scha.schmidt_real(m_ext, n_ext, grid=False))

    for ki, mi, ni, g, sch in zip(k_in, m_in, n_in, gcoefs_in, schmidt_in):
    
        m_abs = abs(mi)
    
        cossin = numpy.cos(m_abs*phiv) if mi >= 0 else numpy.sin(m_abs*phiv)
        sinmcos = numpy.sin(m_abs*phiv) if mi >= 0 else -numpy.cos(m_abs*phiv)

        leg = lpmv(m_abs, ni, numpy.cos(thetav))*sch
        dleg = scha.dlpmv(m_abs, ni, numpy.cos(thetav))*(-numpy.sin(thetav))*sch
    
        x += g*cossin*dleg*(a_r/rv)**(ni+2)
        y += g*sinmcos*m_abs*leg/numpy.sin(thetav)*(a_r/rv)**(ni+2)
        z -= (ni+1)*(g*cossin)*leg*(a_r/rv)**(ni+2)

    for ki, mi, ni, g, sch in zip(k_ext, m_ext, n_ext, gcoefs_ext, schmidt_ext):
    
        m_abs = abs(mi)
    
        cossin = numpy.cos(m_abs*phiv) if mi >= 0 else numpy.sin(m_abs*phiv)
        sinmcos = numpy.sin(m_abs*phiv) if mi >= 0 else -numpy.cos(m_abs*phiv)

        leg = lpmv(m_abs, ni, numpy.cos(thetav))*sch
        dleg = scha.dlpmv(m_abs, ni, numpy.cos(thetav))*(-numpy.sin(thetav))*sch
    
        x += g*cossin*dleg*(rv/a_r)**(ni-1)
        y += g*sinmcos*m_abs*leg/numpy.sin(thetav)*(rv/a_r)**(ni-1)
        z += ni*(g*cossin)*leg*(rv/a_r)**(ni-1)

    return x, y, z

def mehler_field(m_mehler, gcoefs, rv, thetav, phiv, theta_0):

    x = numpy.zeros_like(thetav)
    y = x.copy()
    z = y.copy()
    
    for mm, g in zip(m_mehler, gcoefs):

        mm_abs = numpy.abs(mm)
        
        cossin = cos(mm_abs*phiv) if mm >= 0 else sin(mm_abs*phiv)
        sinmcos = sin(mm_abs*phiv) if mm >= 0 else -cos(mm_abs*phiv)

        mehler_dp = mehler.dmehler_t(mm_abs, 0, thetav, theta_0, normalized=True, mfactor=True)
        mehler_p = mehler.mehler_t(mm_abs, 0, thetav, theta_0, normalized=True, mfactor=True)

        x += r_mehler(rv)*g*mehler_dp * cossin
        y += r_mehler(rv)*g*mm_abs/sin(thetav)*mehler_p*sinmcos
        z += dr_mehler(rv)*g*mehler_p*cossin

    return x, y, z
    
def synth_field(kmn_legendre, m_mehler, gp, rv, thetav, phiv, theta_0):

    #legendre
    gp_legendre = gp[:len(kmn_legendre[0])]
    gp_mehler = gp[len(kmn_legendre[0]):]
    
    x, y, z = scha.xyzfield(*kmn_legendre, gp, thetav, phiv)

    #mehler
    for mm, g in zip(m_mehler, gp_mehler):

        mm_abs = numpy.abs(mm)
        
        cossin = cos(mm_abs*phiv) if mm >= 0 else sin(mm_abs*phiv)
        sinmcos = sin(mm_abs*phiv) if mm >= 0 else -cos(mm_abs*phiv)

        mehler_dp = mehler.dmehler_t(mm_abs, 0, thetav, theta_0, normalized=True, mfactor=True)
        mehler_p = mehler.mehler_t(mm_abs, 0, thetav, theta_0, normalized=True, mfactor=True)

        x += r_mehler(rv)*g*mehler_dp * cossin
        y += r_mehler(rv)*g*mm_abs/sin(thetav)*mehler_p*sinmcos
        z += dr_mehler(rv)*g*mehler_p*cossin #magia chunga

    return x, y, z

def r_mehler(r):
    return numpy.sqrt(a_r/r)*(numpy.log(r/a_r)+2)

def dr_mehler(r):
    return -1/(2*r)*numpy.sqrt(a_r/r)*numpy.log(r/a_r)
