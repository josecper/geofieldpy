import numpy
import scipy

import scha
import rscha_r
import xyzfield
import geofield
import trig
import constants
import bspline

class Model:
    """
    clase que reprensenta un modelo rscha-2d guay
    todos los ángulos están en radianes
    """

    def __init__(self):
        pass

    # paso 1: los parametros
    def set_model_params(self, theta_c, phi_c, theta_0d, cap_edge=0.1,
                         kmax_int=3, kmax_ext=3, m_max=2, g10_ref=-30,
                         knots=numpy.arange(-1000, 2050, 50)):
        self.theta_c = theta_c
        self.phi_c = phi_c
        self.theta_0d = theta_0d
        self.theta_0p = theta_0d + cap_edge
        self.kmax_int = kmax_int
        self.kmax_ext = kmax_ext
        self.m_max = m_max
        self.g10_ref = g10_ref
        self.knots = knots

        # ya se puede calcular el grado
        self.calculate_degrees()

    def calculate_degrees(self):

        maxk = max(self.kmax_int, self.kmax_ext)
        ms = numpy.arange(0, maxk+1)
        roots = scha.degree(ms, theta0=self.theta_0p, max_k=maxk)
        k, m, n = numpy.array(scha.join_roots(roots))
        km_even = ((k-numpy.abs(m)) % 2 == 0) & (k <= maxk)
        k_even, m_even, n_even = k[km_even], m[km_even], n[km_even]

        self.k_int = k_even[k_even <= self.kmax_int]
        self.m_int = m_even[k_even <= self.kmax_int]
        self.n_int = n_even[k_even <= self.kmax_int]

        self.k_ext = k_even[(k_even > 0) & (k_even <= self.kmax_ext)]
        self.m_ext = m_even[(k_even > 0) & (k_even <= self.kmax_ext)]
        self.n_ext = n_even[(k_even > 0) & (k_even <= self.kmax_ext)]
        
        m_mehler = [0]
        for mm in numpy.arange(1, self.m_max+1):
            m_mehler.extend([mm, -mm])
        self.m_mehler = numpy.array(m_mehler)

    # paso 2: la dataaaaaaa
    def add_data(self, fname):
        """
        formato: ref, ref, t, t_err, lat(º), lon(º),
        D(º), I(º), a95(º), F(uT), F_err(uT), misterioso
        """
        datos_t = numpy.loadtxt(fname).T
        # convención ancestral
        datos_t[datos_t == 999] = numpy.nan

        thetav = numpy.deg2rad(90 - datos_t[4])
        phiv = numpy.deg2rad(datos_t[5])

        # trim
        in_cap = trig.angulardist(thetav, phiv,
                                  self.theta_c, self.phi_c) < self.theta_0d

        self.thetav = thetav[in_cap]
        self.phiv = phiv[in_cap]

        self.tv = datos_t[2][in_cap]
        self.D_o = numpy.deg2rad(datos_t[6][in_cap])
        self.I_o = numpy.deg2rad(datos_t[7][in_cap])
        self.F_o = datos_t[9][in_cap]

        # encontrar nans
        self.nan_D = numpy.isnan(self.D_o)
        self.nan_I = numpy.isnan(self.I_o)
        self.nan_F = numpy.isnan(self.F_o)
        self.nan_DIF = numpy.concatenate((self.nan_D,
                                          self.nan_I,
                                          self.nan_F))
        
        # convertir a esferas esféricas
        self.r_geo, self.theta_geo, self.phi_geo = trig.geo2sph(self.thetav,
                                                                self.phiv)
        
        # sintetizar el dipoling
        self.Bx_dip, self.By_dip, self.Bz_dip = self.synth_dipole(self.theta_geo,
                                                                  self.phi_geo)

        self.D_dip, self.I_dip, self.F_dip, self.H_dip = xyzfield.xyz2difh(self.Bx_dip,
                                                                           self.By_dip,
                                                                           self.Bz_dip)

        self.F_dip_avg = numpy.average(self.F_dip)
        # rotar al casquete mágico
        self.r_geo, self.theta_r, self.phi_r = self.rotate_coords_to_cap(self.r_geo,
                                                                         self.theta_geo,
                                                                         self.phi_geo)

        self.Bx_dip_r, self.By_dip_r, self.Bz_dip_r = self.rotate_vectors_to_cap(self.theta_geo,
                                                                                 self.phi_geo,
                                                                                 self.Bx_dip,
                                                                                 self.By_dip,
                                                                                 self.Bz_dip,
                                                                                 self.theta_r)

        # calcular los residuolos
        self.calculate_residuals()

    def solve(self):

        Adift = self.model_matrix(self.tv, self.r_geo, self.theta_r, self.phi_r)[~self.nan_DIF]
        self.Adift_last_solved = Adift
        
        data = numpy.concatenate((self.D_res, self.I_res, self.F_res_norm))[~self.nan_DIF]
        self.data_last_solved = data

        gptlsr = numpy.linalg.lstsq(Adift, data)[0]
        self.g_last_solved = gptlsr

        return gptlsr

    def synth_data(self, tv, rv, thetav, phiv, g=None):

        if g is None:
            g = self.g_last_solved

        rv, theta_r, phi_r = self.rotate_coords_to_cap(rv, thetav, phiv)
        Bx, By, Bz = self.synth_dipole(thetav, phiv)
        D_dip, I_dip, F_dip, H_dip = xyzfield.xyz2difh(Bx, By, Bz)
        
        Bxr, Byr, Bzr = self.rotate_vectors_to_cap(thetav, phiv,
                                                   Bx, By, Bz, theta_r)
        
        adift = self.model_matrix(tv, rv, theta_r, phi_r, Bxr, Byr, Bzr)
        synth = adift @ g

        Ds, Is, Fs = numpy.split(synth, 3)

        D_m = D_dip + Ds
        I_m = I_dip + Is
        F_m = F_dip + Fs*self.F_dip_avg

        return D_m, I_m, F_m
        
    def synth_dipole(self, thetav, phiv):

        Bx_dip, By_dip, Bz_dip = scha.xyzfield((1,), (0,), (1,),
                                               (self.g10_ref,),
                                               thetav, phiv)

        return Bx_dip, By_dip, Bz_dip

    def rotate_coords_to_cap(self, r_geo, theta_geo, phi_geo):

        rot_mat = scha.rotation_matrix(self.theta_c, self.phi_c, invert=True)
        r_geo, theta_r, phi_r = scha.rotate_coords(r_geo,
                                                   theta_geo,
                                                   phi_geo,
                                                   rot_mat)

        return r_geo, theta_r, phi_r

    def rotate_vectors_to_cap(self, thetav, phiv, Bx, By, Bz, theta_r):
        
        Bx_dip_r = numpy.empty_like(Bx)
        By_dip_r = numpy.empty_like(By)
        Bz_dip_r = numpy.empty_like(Bz)
        for i, (xx, yy, zz, th_i, phi_i, th_ri) in enumerate(zip(Bx,
                                                                 By,
                                                                 Bz,
                                                                 thetav,
                                                                 phiv,
                                                                 theta_r)):
            Bx_dip_r[i], By_dip_r[i], Bz_dip_r[i] = scha.rotate_vector(xx, yy, zz,
                                                                       self.theta_c,
                                                                       self.phi_c,
                                                                       th_i, phi_i, th_ri)

        return Bx_dip_r, By_dip_r, Bz_dip_r

    def calculate_residuals(self):

        self.D_res = trig.mindiff(self.D_o, self.D_dip)
        self.I_res = trig.mindiff(self.I_o, self.I_dip)
        self.F_res = self.F_o - self.F_dip
        self.F_res_norm = self.F_res/self.F_dip_avg

    def temporal_matrix(self, tv):
        return bspline.condition_array(self.knots, tv)

    def spatial_matrix(self, rv, thetav, phiv, Bx=None, By=None, Bz=None):

        if (Bx is None) or (By is None) or (Bz is None):
            Bx = self.Bx_dip_r
            By = self.By_dip_r
            Bz = self.Bz_dip_r
        
        Adif_rscha = numpy.concatenate(
            rscha_r.rscha_condition_matrix_dif((self.k_int,
                                                self.m_int,
                                                self.n_int),
                                               (self.k_ext,
                                                self.m_ext,
                                                self.n_ext),
                                               self.m_mehler,
                                               rv, thetav, phiv,
                                               self.theta_0p,
                                               Bx, By, Bz)
            )

        return Adif_rscha

    def model_matrix(self, tv, rv, thetav, phiv, Bx=None, By=None, Bz=None):
        Adif_rscha_test = self.spatial_matrix(rv, thetav, phiv, Bx, By, Bz)
        At = self.temporal_matrix(tv)
        At3 = numpy.vstack((At, At, At))

        Adift = numpy.concatenate([Adif_rscha_test*At3[:, i:i+1]
                                   for i in range(len(self.knots))], axis=1)

        return Adift
