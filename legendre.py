import scipy
import scipy.special
import numpy
import sys

fs=open(sys.argv[1],"r")

theta=float(sys.argv[2])
phi=float(sys.argv[3])
if(len(sys.argv) > 4):
    rparam=float(sys.argv[4])
else:
    rparam=1.0
if(len(sys.argv) > 5):
    lmax = int(sys.argv[5])
else:
    lmax = 13

gcoefs=numpy.zeros((lmax+1,lmax+1))
hcoefs=numpy.zeros((lmax+1,lmax+1))

legendre,dlegendre=scipy.special.lpmn(lmax+1,lmax+1,scipy.cos(theta))

for line in fs.readlines()[5:]:
    elems=line.split()
    l=int(elems[0])
    m=int(elems[1])
    value=float(elems[2])

    if(l <= lmax):
        if(m < 0):
            hcoefs[-m,l]=value
        else:
            gcoefs[m,l]=value

#calcular xyz
x,y,z=0,0,0

for l in range(1,lmax+1):
    for m in range(0,l+1):
        deltax=rparam**(l+2)*(gcoefs[m,l]*scipy.cos(m*phi)+hcoefs[m,l]*scipy.sin(m*phi))*dlegendre[m,l]*(-scipy.sin(theta))
        deltay=rparam**(l+2)*(gcoefs[m,l]*scipy.sin(m*phi)-hcoefs[m,l]*scipy.cos(m*phi))*m*legendre[m,l]/(scipy.sin(theta))
        deltaz=rparam**(l+2)*(l+1)*(gcoefs[m,l]*scipy.cos(m*phi)+hcoefs[m,l]*scipy.sin(m*phi))*legendre[m,l]

        x+=deltax
        y+=deltay
        z+=deltaz
        
        #print("l = {0}, m = {1}, dP/dtheta = {3},   delta = {2}, x = {4}".format(l,m,delta,dlegendre[m,l]*(-scipy.sin(theta)),x))
        print("l = {0}, m = {1}, deltax = {2}, deltay = {3}, deltaz = {4}, x = {5}, y = {6}, z = {7} ".format(l,m,deltax,deltay,deltaz,x,y,z))
