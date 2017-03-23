import numpy

import rscha_r
import field_plots
import fibonacci_sphere
import trig
import coords
from mpl_toolkits.basemap import Basemap
import scha
import xyzfield
import constants
import bspline

import importlib
importlib.reload(scha)

cos = numpy.cos
sin = numpy.sin

datos_t = numpy.loadtxt("../data/rscha2d/dato_synt_sha.dat", skiprows=1).T

tv = datos_t[0]
thetav = numpy.deg2rad(90-datos_t[1])
phiv = numpy.deg2rad(datos_t[2])
D_o = numpy.deg2rad(datos_t[3])
I_o = numpy.deg2rad(datos_t[4])
F_o = datos_t[6]

# conversión a esféricas
r_geo, theta_geo, phi_geo = trig.geo2sph(thetav, phiv)

theta_c, phi_c, theta_0d, theta_0 = numpy.deg2rad((90-42.0, 20.0, 26.0, 50.0))
rot_mat = scha.rotation_matrix(theta_c, phi_c, invert=True)
r_geo, theta_r, phi_r = scha.rotate_coords(r_geo, theta_geo, phi_geo, rot_mat)

# en primera aproximación, a falta de algo más mejor
g01_dip = -30.0
k_dip, m_dip, n_dip = numpy.atleast_1d((1,), (0,), (1,))

# síntesis del dipolo en todos los datas
Bx_dip, By_dip, Bz_dip = scha.xyzfield((1,), (0,), (1,), (g01_dip,),
                                       theta_geo, phi_geo)
Bx_dip_r = numpy.empty_like(Bx_dip)
By_dip_r = numpy.empty_like(By_dip)
Bz_dip_r = numpy.empty_like(Bz_dip)

for i, (xx, yy, zz, th_i, phi_i, th_ri) in enumerate(zip(Bx_dip, By_dip,
                                                         Bz_dip, thetav,
                                                         phiv, theta_r)):
    Bx_dip_r[i], By_dip_r[i], Bz_dip_r[i] = scha.rotate_vector(xx, yy, zz,
                                                               theta_c, phi_c,
                                                               th_i, phi_i,
                                                               th_ri)

D_dip, I_dip, F_dip, H_dip = xyzfield.xyz2difh(Bx_dip, By_dip, Bz_dip)

D_res = trig.mindiff(D_o, D_dip)
I_res = trig.mindiff(I_o, I_dip)
F_res = F_o - F_dip

knots = numpy.arange(-2100, 2050, 50)

At = bspline.condition_array(knots, tv)

theta_0p = theta_0d + 0.1

ms = numpy.arange(0, 4)
roots = scha.degree(ms, theta0=theta_0p, max_k=3)
k, m, n = scha.join_roots(roots)

k, m, n = numpy.array((k, m, n))
km_even = ((k-numpy.abs(m)) % 2 == 0) & (k <= 3)
k_even, m_even, n_even = k[km_even], m[km_even], n[km_even]
m_mehler= numpy.array((0, 1, -1, 2, -2))

for ki, mi, ni in zip(k,m,n):
    print(f"{ki:<10} {mi:<10} {ni:<10}")

#super mega ultra condition matrix:
Adif_rscha_test = numpy.concatenate(rscha_r.rscha_condition_matrix_dif((k_even, m_even, n_even),
                                                     (k_even[1:], m_even[1:], n_even[1:]),
                                                     m_mehler,
                                                     r_geo, theta_r, phi_r, theta_0p,
                                                     Bx_dip_r, By_dip_r, Bz_dip_r))

At3 = numpy.vstack((At, At, At))
Adift = numpy.concatenate([Adif_rscha_test*At3[:, i:i+1] for i in range(len(knots))], axis=1)
#puede ser correcto bien

F_dip_avg=numpy.average(F_dip)
F_res_norm = F_res/F_dip_avg

gptlsr = numpy.linalg.lstsq(Adift, numpy.concatenate((D_res, I_res, F_res/F_dip_avg)))[0]

times_long = numpy.linspace(-2000, 1900, 20)

theta_in, phi_in = fibonacci_sphere.grid(n=7500)
in_cap = trig.angulardist(theta_in, phi_in, theta_c, phi_c) < theta_0p
theta_in = theta_in[in_cap]
phi_in = phi_in[in_cap]

lon_c, lat_c = numpy.rad2deg((phi_c, numpy.pi/2 - theta_c))
base_cap = Basemap(projection="npaeqd", lon_0 = 0, boundinglat=90-numpy.rad2deg(theta_0p))
base_world = Basemap(projection="aeqd", lon_0 = lon_c, lat_0 = lat_c, lat_ts=45.0,
                    width=base_cap.xmax, height=base_cap.ymax, resolution="l")

for time in times_long:

    timev = numpy.atleast_1d(time)
    times_d, r_d, theta_d, phi_d = coords.anything(numpy.ones_like(theta_in)*constants.a_r,
                                                   theta_in,
                                                   phi_in,
                                                   timev)

    r_dg, theta_dg, phi_dg = trig.geo2sph(theta_d, phi_d)
    r_dgr, theta_dgr, phi_dgr = scha.rotate_coords(r_dg, theta_dg, phi_dg, rot_mat)
    Bx_dip_d, By_dip_d, Bz_dip_d = scha.xyzfield(k_dip,m_dip,n_dip, (g01_dip,), theta_dg, phi_dg)

    Bx_dip_dr = numpy.empty_like(Bx_dip_d); By_dip_dr = Bx_dip_dr.copy(); Bz_dip_dr = Bx_dip_dr.copy()

    for i, (xx, yy, zz, th_i, phi_i, th_ri) in enumerate(zip(Bx_dip_d,By_dip_d,Bz_dip_d,theta_dg,phi_dg,theta_dgr)):
        Bx_dip_dr[i], By_dip_dr[i], Bz_dip_dr[i] = scha.rotate_vector(xx, yy, zz, theta_c, phi_c,
                                                                      th_i, phi_i, th_ri)    

    Adif_syn = numpy.concatenate(rscha_r.rscha_condition_matrix_dif((k_even, m_even, n_even),
                                                         (k_even[1:], m_even[1:], n_even[1:]),
                                                         m_mehler,
                                                         r_dgr, theta_dgr, phi_dgr, theta_0p,
                                                         Bx_dip_dr, By_dip_dr, Bz_dip_dr))

    A3t_syn = numpy.vstack([bspline.condition_array(knots, times_d)]*3)
    Adift_syn = numpy.concatenate([Adif_syn*A3t_syn[:, i:i+1] for i in range(len(knots))], axis=1)

    del Adif_syn, A3t_syn

    synth_d = Adift_syn @ gptlsr
    Ds_d, Is_d, Fs_d = numpy.split(synth_d, 3)

    D_dip_d, I_dip_d, F_dip_d, H_dip_d = xyzfield.xyz2difh(Bx_dip_d, By_dip_d, Bz_dip_d)

    D_m = D_dip_d + Ds_d
    I_m = I_dip_d + Is_d
    F_m = Fs_d*F_dip_avg + F_dip_d
    
    #fig = field_plots.component_residual_plot(theta_dgr, phi_dgr, theta_0p, theta_c, phi_c,
    #                (numpy.rad2deg(D_m), numpy.rad2deg(I_m), F_m),
    #                scales=("symmetric", "positive", "positive"),
    #                cmaps=("Spectral", "Spectral", "Spectral"),
    #                lines=True)
    #fig.savefig('../data/rscha2d/mapas/rscha2d_{time:d}.jpg'.format(time=int(time)))
    #print("do a heck {time}".format(time=time))

    print(time)
    
    field_plots.super_video_frame(theta_dgr, phi_dgr, theta_0p, theta_c, phi_c,
                                  (numpy.rad2deg(D_m), numpy.rad2deg(I_m), F_m),
                                  base_cap, base_world,
                                  '../data/rscha2d/mapas/rscha2d_{time:d}.jpg'.format(time=int(time)),
                                  scales=("symmetric", "positive", "positive"),
                                  cmaps=("Spectral", "Spectral", "Spectral"),
                                  lines = True)
    
