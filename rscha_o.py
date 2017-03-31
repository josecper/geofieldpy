import scha
import trig
import xyzfield
import constants
import numpy
import rscha_r
import bspline

class Model:

    # todo está en radianes por la gloria de tu madre jarl

    def __init__(self):
        self.data = None

    def set_cap(self, theta_c, phi_c, theta_0, dtheta_pad):

        # centro del casquete (colat, lon)
        self.theta_c = theta_c
        self.phi_c = phi_c

        # radio (colat desde el centro)
        self.theta_0 = theta_0
        
        # extra para evitar efectos de borde
        self.theta_0p = theta_0 + dtheta_pad

        # precalcular matriz de rotación para después
        self.rot_mat = scha.rotation_matrix(theta_c, phi_c, invert=True)

    def set_model_parameters(self, k_int_max, k_ext_max, m_max, knots, g01_ref):

        self.k_int_max = k_int_max
        self.k_ext_max = k_ext_max
        self.g01_ref = g01_ref
        self.knots = knots
        self.m_max = m_max

    def set_regularization_parameters(self, spatial_reg, temporal_reg):
        self.spatial_reg = spatial_reg
        self.temporal_reg = temporal_reg

    def crop_cap(self, thetav, phiv):
        
        # har har har cap
        in_cap = trig.angulardist(thetav, phiv, self.theta_c, self.phi_c) < self.theta_0
        return in_cap

    def add_data(self, t, thetav, phiv, D, I, F, crop = True, to_spherical = True):

        # crop the it
        if crop:
            in_cap = self.crop_cap(thetav, phiv)

        # convertir a esféricas
        if to_spherical:
            r_sph, theta_sph, phi_sph = trig.geo2sph(thetav, phiv)
        else:
            r_sph = numpy.ones_like(thetav)*constants.a_r
            theta_sph, phi_sph = thetav, phiv

        self.t = t
        self.r_sph = r_sph
        self.theta_sph = theta_sph
        self.phi_sph = phi_sph
        self.D = D
        self.I = I
        self.F = F

    def calculate_ref_dipole(self):

        self.Bx_dip_sph, self.By_dip_sph, self.Bz_dip_sph = scha.xyzfield((1,), (0,), (1,),
                                                                          (self.g01_ref,),
                                                                          self.theta_sph,
                                                                          self.phi_sph)

        self.D_dip, self.I_dip, self.F_dip, self.H_dip = xyzfield.xyz2difh(self.Bx_dip_sph,
                                                                           self.By_dip_sph,
                                                                           self.Bz_dip_sph)

        self.F_dip_avg = numpy.average(self.F_dip)

    def calculate_delta(self):

        self.D_res = self.D - self.D_dip
        self.I_res = self.I - self.I_dip
        self.F_res = self.F - self.F_res
        self.F_res_norm = self.F_res / self.F_dip_avg
        
    def rotate_to_cap(self):

        # rotate a very big rock (earth)
        self.r_r, self.theta_r, self.phi_r = scha.rotate_coords(self.r_sph,
                                                                self.theta_sph,
                                                                self.phi_sph,
                                                                self.rot_mat)

        # do a rotate to some vector (dipolo's)
        self.Bx_dip_r = numpy.zeros_like(self.Bx_dip_sph)
        self.By_dip_r = self.Bx_dip_r.copy()
        self.Bz_dip_r = self.Bx_dip_r.copy()

        for i, (xx, yy, zz, th_i, phi_i, th_ri) in enumerate(zip(self.Bx_dip_sph,self.By_dip_sph,
                                                                 self.Bz_dip_sph,
                                                                 self.theta_sph,self.phi_sph,
                                                                 self.theta_r)):
            Bx_dip_r[i], By_dip_r[i], Bz_dip_r[i] = scha.rotate_vector(xx, yy, zz,
                                                                       self.theta_c, self.phi_c,
                                                                       th_i, phi_i, th_ri)

    def calculate_condition_matrix(self):

        Adif_syn = numpy.concatenate(rscha_r.rscha_condition_matrix_dif(self.kmn_in,
                                                                        self.kmn_ext,
                                                                        self.m_mehler,
                                                                        self.r_r,
                                                                        self.theta_r,
                                                                        self.phi_r,
                                                                        self.theta_0p,
                                                                        self.Bx_dip_r,
                                                                        self.By_dip_r,
                                                                        self.Bz_dip_r))
        A3t_syn = numpy.vstack([bspline.condition_array(self.knots, self.t)]*3)
        self.Adift_syn = numpy.concatenate([Adif_syn * A3t_syn[:, i:i+1] for i in range(len(knots))],
                                           axis = 1)
        

    def calculate_degrees(self, debug=True):

        k_maximum = max(self.k_ext_max, self.k_int_max)
        ms = numpy.arange(0, k_maximum + 1)
        k, m, n = scha.join_roots(scha.degree(ms, theta0 = self.theta_0p, max_k = k_maximum))
        k, m, n = numpy.array((k,m,n))
        
        km_even = ((k-numpy.abs(m)) % 2 == 0) & (k <= 3)
        k_even, m_even, n_even = k[km_even], m[km_even], n[km_even]
        m_mehler = [0]
        for i in range(1, self.m_max+1): m_mehler.extend([i, -i])

        self.kmn_in = k_even, m_even, n_even
        self.kmn_ext = k_even[1:], m_even[1:], n_even[1:]
        self.m_mehler = numpy.array(m_mehler)

        if debug:
            print("Internal field")
            print("{:<10} {:<10} {:<10}".format("k", "m", "n"))
            print("-"*32)
            for ki, mi, ni in zip(*self.kmn_in):
                print(f"{ki:10f} {mi:10f} {ni:10f}")
            print("-"*32)
            print("External field")
            print("{:<10} {:<10} {:<10}".format("k", "m", "n"))
            print("-"*32)
            for ki, mi, ni in zip(*self.kmn_ext):
                print(f"{ki:10f} {mi:10f} {ni:10f}")
            print("Mehler field")
            print("{:<10} ".format("m"))
            print("-"*32)
            for mi in self.m_mehler:
                print(f"{mi:10f}")

    def solve(self):
        # not yet :(
        pass

    
