import numpy
import scipy, scipy.special, scipy.misc

def invert(coords ,data, order=13, rparam=1.0):

    #todo: regular grid
    thetav,phiv=coords
    
    if len(data) == 1:
        #assume x ... y ... z ...
        data = numpy.array(data)
        ndata = data.shape[-1]//3
    elif len(data) == 3:
        #assume (x,y,z)
        data = numpy.concatenate(data)
        ndata = data.shape[0] // 3
    else: raise Exception("invalid data shape")

    #order array (same as CHAOS)
    orders=[]
    for l in range(1,order+1):
        for m in range(0,l+1):
            if m == 0:
                orders.append((m,l))
            else:
                orders.append((m,l))
                orders.append((-m,l))

    #legendre polynomials
    leg=numpy.zeros((len(thetav),order+1,order+1))
    dleg=leg.copy()

    costhetav=scipy.cos(thetav)
    
    for cth,i in zip(costhetav,range(len(costhetav))):
        #indexed as m,l
        leg[i,:,:], dleg[i,:,:] = scipy.special.lpmn(order,order,cth)

    #stack them in a single thing + normalize
    def schmidt(m,l):
        return (-1)**m*scipy.sqrt((2-(m==0))*scipy.misc.factorial(l-m)/scipy.misc.factorial(l+m))

    legs=[]; dlegs=[]
    
    for m,l in orders:
        legs.append(schmidt(m,l)*leg[:,abs(m),l])
        dlegs.append(schmidt(m,l)*dleg[:,abs(m),l])

    legs=numpy.array(legs)
    dlegs=numpy.array(dlegs)
    #debuge ~~~~
    #return legs,dlegs
    
    #build condition array A
    pass
    
    
