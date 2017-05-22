import numpy

cos = numpy.cos
sin = numpy.sin

a_ellip = 6371.137
b_ellip = 6356.752

def angulardist(theta1, phi1, theta2, phi2):
    return numpy.arccos(sin(theta1)*sin(theta2)*cos(phi1 - phi2)+cos(theta1)*cos(theta2))

def center_angles(angles):
    angles = numpy.asarray(angles)
    return numpy.mod(angles+numpy.pi, 2*numpy.pi) - numpy.pi

def mindiff(a, b):
    return numpy.arctan2(sin(a-b), cos(a-b))

def geo2sph(thetav, phiv):
    theta_s = numpy.arctan2(a_ellip**2/b_ellip**2*sin(thetav), cos(thetav))
    r_s = numpy.sqrt((a_ellip**4*sin(thetav)**2 + b_ellip**4*cos(thetav)**2)/\
                     (a_ellip**2*sin(thetav)**2 + b_ellip**2*cos(thetav)**2))

    return r_s, theta_s, phiv


def relocate(D, I, F, theta, phi, theta_ref, phi_ref):

    lat = numpy.pi/2 - theta
    lon = phi

    lat_gsite = numpy.arctan(numpy.tan(I)/2)
    lat_vgp = numpy.arcsin(numpy.sin(lat_gsite)*numpy.sin(lat)
                           + numpy.cos(lat_gsite)*numpy.cos(lat)
                           * numpy.cos(D))

    beta = numpy.arcsin(numpy.sin(D)*numpy.cos(lat_gsite)
                        / numpy.cos(lat_vgp))

    lon_vgp = numpy.zeros_like(lat_vgp)
    big = (numpy.sin(lat_vgp)*numpy.sin(lat)) >= numpy.sin(lat_gsite)
    lon_vgp[big] = numpy.pi + lon[big] - beta[big]
    lon_vgp[~big] = lon[~big] + beta[~big]

    lat_at = numpy.pi/2 - theta_ref
    lon_at = phi_ref

    lat_gat = numpy.arcsin(numpy.sin(lat_at)*numpy.sin(lat_vgp)
                           + numpy.cos(lat_at)*numpy.cos(lat_vgp) *
                           numpy.cos(lon_at - lon_vgp))

    I_reloc = numpy.arctan(2*numpy.tan(lat_gat))
    D_reloc = numpy.arcsin((numpy.sin(lon_vgp - lon_at)*numpy.cos(lat_vgp))
                           / numpy.cos(lat_at))
    F_reloc = (F*numpy.sqrt(1+3*numpy.sin(lat_at)**2)
               / numpy.sqrt(1+3*numpy.sin(lat)**2))

    return D_reloc, I_reloc, F_reloc
