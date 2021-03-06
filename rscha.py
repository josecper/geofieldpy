import numpy, scipy
import mehler
import scha
import xyzfield
import sys

from numpy import cos, sin
from scipy.special import lpmv
from scha import condition_matrix_dif

a_ellip = 6371.137
b_ellip = 6356.752

def mehler_condition_matrix_xyz(thetav, phiv, m, theta_0):
    """
    Mehler functions base condition matrix [(3*ndata) x ncoefs]. This is to be
    concatenated to the usual Legendre condition matrix to solve the inverse
    problem.
    """

    #esto es la parte que va multiplicada por G, H.
    
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
        
        Ax[:,i] = mehler_dp*cossin
        Ay[:,i] = mm_abs/sin(thetav)*mehler_p*sinmcos
        #Az[:,i] = -0.5*mehler_p*cossin #magia chunga

    Axyz = numpy.concatenate ((Ax,Ay,Az), axis=0)
    return Axyz

#ha ha whoops you died oh well

def norm_leg_even(k, m, n, theta_0):

    m_abs = numpy.abs(m)
    schmidt = scha.schmidt_real(m_abs, n, grid=False)

    return (-sin(theta_0))**2*lpmv(m_abs, n, cos(theta_0))*scha.dndlpmv(m_abs, n, cos(theta_0))*schmidt**2\
        /(2*n+1)

def rscha_spatial_reg_diag(kmn_legendre, m_mehler, theta_0):
    """
    Legendre/Mehler functions spatial regularization matrix (diagonal?? hmm), 
    based on the <B²> norm.
    """
    k_even, m_even, n_even = kmn_legendre
    
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
    
    
def synth_field(kmn_legendre, m_mehler, gp, thetav, phiv, theta_0):

    #legendre
    gp_legendre = gp[:len(kmn_legendre[0])]
    gp_mehler = gp[len(kmn_legendre[0]):]
    
    x, y, z = scha.xyzfield(*kmn_legendre, gp, thetav, phiv)

    #mehler
    for mm, g in zip(m_mehler, gp_mehler):

        mm_abs = numpy.abs(mm)
        
        cossin = cos(mm_abs*phiv) if mm >= 0 else sin(mm_abs*phiv)
        sinmcos = sin(mm_abs*phiv) if mm >= 0 else -cos(mm_abs*phiv)

        #mehler_dp = mehler.dmehler_t(mm_abs, 0, thetav, theta_0, normalized=False) #MAGIC
        mehler_dp = mehler.dmehler_t(mm_abs, 0, thetav, theta_0, normalized=True, mfactor=True)
        mehler_p = mehler.mehler_t(mm_abs, 0, thetav, theta_0, normalized=True, mfactor=True)

        x += g*mehler_dp * cossin
        y += g*mm_abs/sin(thetav)*mehler_p*sinmcos
        #z -= 0.5*g*mehler_p*cossin #magia chunga

    return x, y, z
