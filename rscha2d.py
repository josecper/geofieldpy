import numpy
import scipy

import scha
import rscha_r
import xyzfield
import geofield
import trig
import constants
import bspline
import ops

import joblib
import tempfile

memory = joblib.Memory(cachedir=tempfile.mkdtemp())


class Model:
    """
    clase que reprensenta un modelo rscha-2d guay
    todos los ángulos están en radianes pero el archivo está
    en grados ¯\_(ツ)_/¯
    """

    def __init__(self):
        #self.model_matrix = memory.cache(self.model_matrix)
        self.spatial_matrix = memory.cache(self.spatial_matrix)
        pass

    # paso 1: los parametros
    def set_model_params(self, theta_c, phi_c, theta_0d, cap_edge=0.1,
                         kmax_int=3, kmax_ext=3, m_max=2, g10_ref=-30,
                         knots=numpy.arange(-1000, 2050, 50),
                         spatial_reg=0.0, temporal_reg=0.0):
        self.theta_c = theta_c
        self.phi_c = phi_c
        self.theta_0d = theta_0d
        self.theta_0p = theta_0d + cap_edge
        self.kmax_int = kmax_int
        self.kmax_ext = kmax_ext
        self.m_max = m_max
        self.g10_ref = g10_ref
        self.knots_ok = numpy.concatenate((3*[knots[0]], knots))
        self.knots = numpy.concatenate((3*[knots[0]], knots))[:-1]
        self.spatial_reg = spatial_reg
        self.temporal_reg = temporal_reg

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

    def normalize(self):
        pass
        
    # paso 2: la data
    def add_data(self, fname):
        """
        formato: ref, ref, t, t_err, lat(º), lon(º),
        D(º), I(º), a95(º), F(uT), F_err(uT), arq_vol
        """
        datos_t = numpy.loadtxt(fname).T
        self.volcanic = (datos_t[0] == 888)
        print(self.volcanic.sum())
        # convención ancestral
        datos_t[datos_t == 999] = numpy.nan

        thetav = numpy.deg2rad(90 - datos_t[4])
        phiv = numpy.deg2rad(datos_t[5])

        # trim
        in_cap = trig.angulardist(thetav, phiv,
                                  self.theta_c, self.phi_c) < self.theta_0d

        self.ids = numpy.arange(len(thetav))[in_cap]
        self.in_cap = in_cap
        self.volcanic = self.volcanic[in_cap]

        self.lat = datos_t[4][in_cap]
        self.lon = datos_t[5][in_cap]

        self.thetav = thetav[in_cap]
        self.phiv = phiv[in_cap]

        self.tv = datos_t[2][in_cap]
        self.t_err = datos_t[3][in_cap]
        self.D_o = numpy.deg2rad(datos_t[6][in_cap])
        self.I_o = numpy.deg2rad(datos_t[7][in_cap])
        self.a95 = numpy.deg2rad(datos_t[8][in_cap])

        # normalized form???
        self.D_o = trig.mindiff(self.D_o, 0)
        self.I_o = trig.mindiff(self.I_o, 0)
        
        self.F_o = datos_t[9][in_cap]
        self.F_err = datos_t[10][in_cap]

        # encontrar nans
        self.nan_D = numpy.isnan(self.D_o)
        self.nan_I = numpy.isnan(self.I_o)
        self.nan_F = numpy.isnan(self.F_o)
        self.nan_DIF = numpy.concatenate((self.nan_D,
                                          self.nan_I,
                                          self.nan_F))

        self.refresh_data()

    def refresh_data(self):
        
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

    def find_outliers(self, g=None):

        if g is None:
            g = self.g_last_solved

        sigma_65_D = 81/140*self.a95/numpy.cos(self.I_o)
        sigma_65_I = 81/140*self.a95
        sigma_65_F = self.F_err.copy()

        D_rem_bd, I_rem_bd, F_rem_bd = self.synth_data(self.tv, self.r_geo,
                                                       self.theta_geo,
                                                       self.phi_geo, g)

        out_D = numpy.abs(trig.mindiff(D_rem_bd, self.D_o)) > 3*sigma_65_D
        out_I = numpy.abs(trig.mindiff(I_rem_bd, self.I_o)) > 3*sigma_65_I
        out_F = numpy.abs((F_rem_bd - self.F_o)) > 3*sigma_65_F

        return out_D, out_I, out_F

    def wiggle(self):

        sigma_65_D = 81/140*self.a95/numpy.cos(self.I_o)
        sigma_65_I = 81/140*self.a95
        sigma_65_F = self.F_err.copy()
        t_err = self.t_err.copy()

        sigma_65_D[numpy.isnan(sigma_65_D) |
                   (self.a95 < constants.min_a95)] = 81/140*constants.min_a95
        sigma_65_I[numpy.isnan(sigma_65_I) |
                   (self.a95 < constants.min_a95)] = 81/140*constants.min_a95

        sigma_65_F[numpy.isnan(sigma_65_F) |
                   (sigma_65_F < constants.min_Ferr)] = constants.min_Ferr

        t_err[t_err < 1] = 1
        
        self.D_o = self.D_o + sigma_65_D*numpy.random.randn(*self.D_o.shape)
        self.I_o = self.I_o + sigma_65_I*numpy.random.randn(*self.I_o.shape)
        self.F_o = self.F_o + sigma_65_F*numpy.random.randn(*self.F_o.shape)

        self.D_o = trig.mindiff(self.D_o, 0)
        self.I_o = trig.mindiff(self.I_o, 0)

        self.tv = self.tv + (numpy.random.random(*self.tv.shape) - 0.5)*(2*self.t_err)

    def change_indices(Adift):
        pass

    def solve(self, complete=False):

        n_knots = len(self.knots)
        n_degrees = (len(self.k_int)
                     + len(self.k_ext)
                     + len(self.m_mehler))

        Adift = self.model_matrix(self.tv, self.r_geo, self.theta_r, self.phi_r,
                                  reverse_order=True)[~self.nan_DIF] # CUIDAO!!!!!!!!!!!!!!!!!!!!!!!
        self.Adift_last_solved = Adift

        data = numpy.concatenate((self.D_res, self.I_res, self.F_res_norm))[~self.nan_DIF]
        self.data_last_solved = data

        if (self.temporal_reg == 0.0) and (self.spatial_reg == 0.0):
            gptlsr = numpy.linalg.lstsq(Adift, data)[0]
        else:
            M_reg = self.regularization_matrix()
            # how_big = ops.rms(Adift.T @ Adift), ops.rms(M_reg)
            # print(f"let's HECK: {how_big}")
            gptlsr_reversed = numpy.linalg.lstsq(Adift.T @ Adift + M_reg,
                                                 Adift.T @ data)[0]

        gptlsr = numpy.concatenate(gptlsr_reversed.reshape(n_degrees,n_knots).T)
        self.g_last_solved = gptlsr

        if complete:
            ds = numpy.diag(numpy.repeat(self.reg_br2(), len(self.knots)))
            dtds = (numpy.tile(self.reg_dt2_2(), (n_degrees, n_degrees)) *
                    numpy.repeat(
                        numpy.repeat(
                            numpy.diag(self.reg_br2()),
                            len(self.knots), axis=0), len(self.knots), axis=1))
        
            return {"coefs" : gptlsr,
                    "coefs_reversed" : gptlsr_reversed,
                    "norm_t" : gptlsr_reversed @ dtds @ gptlsr_reversed,
                    "norm_s" : gptlsr_reversed @ ds @ gptlsr_reversed}
        else:
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
        F_m = F_dip + Fs*F_dip

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
        self.F_res_norm = self.F_res/self.F_dip

    def temporal_matrix(self, tv):

        A = bspline.deboor_base(self.knots_ok, tv, 3)[:, :-1]
        #numpy.savetxt("spline.txt", A)
        return A

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
                                               Bx, By, Bz, self.F_dip_avg)
            )

        return Adif_rscha

    def model_matrix(self, tv, rv, thetav, phiv, Bx=None, By=None, Bz=None, reverse_order=False):

        #print("calling model_matrix ...")
        Adif_rscha_test = self.spatial_matrix(rv, thetav, phiv, Bx, By, Bz)
        At = self.temporal_matrix(tv)
        At3 = numpy.vstack((At, At, At))

        if reverse_order:
            Adift = numpy.concatenate([Adif_rscha_test[:, i:i+1]*At3 for i in
                                       range(Adif_rscha_test.shape[1])], axis=1)
        else:
            Adift = numpy.concatenate([Adif_rscha_test*At3[:, i:i+1]
                                       for i in range(len(self.knots))], axis=1)

        return Adift

    # welcome to spaghetti hell
    def regularization_matrix(self):
        rt = self.temporal_reg
        rs = self.spatial_reg

        # dummy to see if it works:

        n_knots = len(self.knots)
        d2 = numpy.diff(numpy.eye(n_knots), 2, axis=0)
        n_degrees = (len(self.k_int)
                     + len(self.k_ext)
                     + len(self.m_mehler))

        d2 = numpy.repeat(
             numpy.repeat(d2.T @ d2, n_degrees, axis=0),
             n_degrees, axis=1)

        #ds = rscha_r.rscha_spatial_reg_diag((self.k_int, self.m_int, self.n_int),
        #                                    (self.k_ext, self.m_ext, self.n_ext),
        #                                    self.m_mehler, self.theta_0p)

        # ds = numpy.diag(numpy.tile(self.reg_br2(), len(self.knots))) (antes)
        ds = numpy.diag(numpy.repeat(self.reg_br2(), len(self.knots))) # CUIDAO

        dtds = (numpy.tile(self.reg_dt2_2(), (n_degrees, n_degrees)) *
                numpy.repeat(
                    numpy.repeat(
                        numpy.diag(self.reg_br2()),
                        len(self.knots), axis=0), len(self.knots), axis=1))
        
        #numpy.savetxt("Mf.txt", dsl * ds)
        #numpy.savetxt("dt2.txt", dtds)
        return rt * (dtds) + rs * ds
        #return rt * (ds * d2) + rs * ds

    def reg_dt2(self):

        n_knots = len(self.knots)
        n_degrees = (len(self.k_int)
                     + len(self.k_ext)
                     + len(self.m_mehler))

        ds = numpy.diff(numpy.eye(n_knots), 2, axis=0)
        ds = ds.T @ ds

        dsl = [numpy.tile(row_ds, (n_degrees, n_degrees)) for row_ds in ds]
        dsl = numpy.vstack(dsl)

        #numpy.savetxt("dsl.txt", dsl)
        return dsl

    def reg_dt2_2(self):

        n_knots = len(self.knots)
        n_degrees = (len(self.k_int)
                     + len(self.k_ext)
                     + len(self.m_mehler))

        ds = numpy.diff(numpy.eye(n_knots), 2, axis=0)
        ds = ds.T @ ds
        
        return ds

    def reg_br2(self):

        cos = numpy.cos
        sin = numpy.sin

        reg_leg_in = numpy.zeros_like(self.k_int)

        for i, (k, m, n) in enumerate(zip(self.k_int, self.m_int, self.n_int)):
            m_abs = abs(m)
            a = 1/(1-cos(self.theta_0p)) if m == 0 else 1/(2-2*cos(self.theta_0p))
            A = -(n+1)**2 * (n+2)**2 / (2*n+1)
            B = a*sin(self.theta_0p)
            C = scha.lpmv(m_abs, n, cos(self.theta_0p))*scha.schmidt_real(m_abs, n)
            D = scha.dndlpmv(m_abs, n, cos(self.theta_0p))*(-sin(self.theta_0p))*scha.schmidt_real(m_abs, n)
            E = 1.

            reg_leg_in[i] = A*B*C*D*E

        reg_leg_ext = numpy.zeros_like(self.k_ext)

        for i, (k, m, n) in enumerate(zip(self.k_ext, self.m_ext, self.n_ext)):
            m_abs = abs(m)
            a = 1/(1-cos(self.theta_0p)) if m == 0 else 1/(2-2*cos(self.theta_0p))
            A = -n**2 * (n-1)**2 / (2*n+1)
            B = a*sin(self.theta_0p)
            C = scha.lpmv(m_abs, n, cos(self.theta_0p))*scha.schmidt_real(m_abs, n)
            D = scha.dndlpmv(m_abs, n, cos(self.theta_0p))*(-sin(self.theta_0p))*scha.schmidt_real(m_abs, n)
            E = 1.

            reg_leg_ext[i] = A*B*C*D*E

        reg_meh = numpy.zeros_like(self.m_mehler, dtype="float")
        reg_meh[0] = 1./(1-cos(self.theta_0p))
        reg_meh[1:] = 1./(2-2*cos(self.theta_0p))

        return numpy.concatenate((reg_leg_in, reg_leg_ext, 5*reg_meh))
