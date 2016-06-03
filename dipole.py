import numpy
import newleg

def ecc_dipole(gcomp, a=6371.0):

    g10,g11,h11,g20,g21,h21,g22,h22 = gcomp[... , :8].transpose()

    #big meme that give r and shit

    L0 = 2*g10*g20 + numpy.sqrt(3)*(g11*g21 + h11+h21)
    L1 = -g11*g20 + numpy.sqrt(3)*(g10*g21 + g11*g22 + h11*h22)
    L2 = -h11*g20 + numpy.sqrt(3)*(g10*h21 + g11*h22 - h11*g22)
    m_2 = g10**2 + g11**2 + h11**2
    E = (L0*g10 + L1*g11 + L2*h11)/(4*m_2)

    xc = a*(L1 - E*g11)/(3*m_2)
    yc = a*(L2 - E*h11)/(3*m_2)
    zc = a*(L0 - E*g10)/(3*m_2)

    return xc, yc, zc

def power(gcomp, lmin, lmax, rparam=1.0, separated=False):

    m,l=newleg.degrees(lmax,start=1)
    powers=rparam**(2*l[l >= lmin]+4)*(l[l >= lmin] + 1)*gcomp[:, numpy.where(l >= lmin)]**2

    if separated:
        return powers.sum(axis=1)
    else:
        return powers.sum(axis=2).sum(axis=1)

def xyz2cyl(x,y,z):

    s = numpy.sqrt(x**2+y**2)
    phi = numpy.arctan2(y,x)

    return s,z,phi
