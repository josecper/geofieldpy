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
