import numpy
import scipy
import scha
import bspline
import xyzfield
import memory_profiler
import gc

@memory_profiler.profile
def do_the_thing():
    cos = numpy.cos; sin = numpy.sin

    super_datos = numpy.loadtxt("/home/josecper/Programs/data/scha/pruebas/archeo019.csv", skiprows=2, delimiter=",", usecols=(0, 3, 5, 6, 8, 9))

    def angulardist(theta1, phi1, theta2, phi2):
        return numpy.arccos(sin(theta1)*sin(theta2)*cos(phi1 - phi2)+cos(theta1)*cos(theta2))

    super_datos[(super_datos == -999) | (super_datos == 999)] = numpy.nan

    theta_c, phi_c, theta_0d, theta_0 = numpy.deg2rad((90-45.0, 15.0, 20.0, 20.0))

    lat, lon = (super_datos.T[-2], super_datos.T[-1])
    super_datos = super_datos[~numpy.isnan(lat) & ~numpy.isnan(lon)]

    thetav, phiv = numpy.deg2rad((90-super_datos.T[4], super_datos.T[5]))
    inside_cap = angulardist(thetav, phiv, theta_c, phi_c) < theta_0d
    super_datos = super_datos[inside_cap]

    thetav, phiv = numpy.deg2rad((90-super_datos.T[4], super_datos.T[5]))
    times, F_orig, D_orig, I_orig = super_datos.T[:4]

    D_orig, I_orig = numpy.deg2rad((D_orig, I_orig))

    del super_datos

    rot_mat = scha.rotation_matrix(theta_c, phi_c, invert=True)
    r, theta_r, phi_r = scha.rotate_coords(1.0, thetav, phiv, rot_mat)

    D_rot = scha.rotate_declination(D_orig, theta_c, phi_c, thetav, phiv, theta_r)
    
    ms = numpy.arange(0, 8);
    roots = scha.degree(ms, theta0 = theta_0, max_k = 7)
    k, m, n = scha.join_roots(roots)
    
    knots = numpy.linspace(-1500, 2500, 10)

    gp = scha.invert_dift(theta_r, phi_r, times, D_rot, I_orig, F_orig, (k, m, n), knots, g0=None, steps=1)

    del gp
    gc.collect()

if __name__ == "__main__":
    do_the_thing()
    print("im going insane")
