import numpy

cos = numpy.cos
sin = numpy.sin

def angulardist(theta1, phi1, theta2, phi2):
    return numpy.arccos(sin(theta1)*sin(theta2)*cos(phi1 - phi2)+cos(theta1)*cos(theta2))

def center_angles(angles):
    angles = numpy.asarray(angles)
    return numpy.mod(angles+numpy.pi, 2*numpy.pi) - numpy.pi
