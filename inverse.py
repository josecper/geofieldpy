import numpy, numpy.linalg
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
    sinthetav=scipy.sin(thetav)
    
    for cth,i in zip(costhetav,range(len(costhetav))):
        #indexed as m,l
        leg[i,:,:], dleg[i,:,:] = scipy.special.lpmn(order,order,cth)

    #stack them in a single thing + normalize
    def schmidt(m,l):
        return (-1)**m*scipy.sqrt((2-(m==0))*scipy.misc.factorial(l-m)/scipy.misc.factorial(l+m))

    legs=[]; dlegs=[]; ms=[]; lplusones=[]; cossin=[]; sinmcos=[]

    #can optimize with le outer product nice
    for m,l in orders:
        legs.append(schmidt(abs(m),l)*leg[:,abs(m),l])
        dlegs.append(schmidt(abs(m),l)*dleg[:,abs(m),l])
        ms.append(abs(m))
        lplusones.append(l+1)
        cossin.append(numpy.cos(m*phiv) if m >= 0 else numpy.sin(-m*phiv)) #-m because yes
        sinmcos.append(numpy.sin(m*phiv) if m >= 0 else -numpy.cos(-m*phiv))

    #axis: 0 = coords, 1 = order
    legs=numpy.array(legs).transpose() #shape = (n, l(l+2))
    dlegs=numpy.array(dlegs).transpose()
    ms=numpy.array(ms)[numpy.newaxis, :] # shape = l(l+2)
    lplusones=numpy.array(lplusones)[numpy.newaxis, :]
    cossin=numpy.array(cossin).transpose() #shape = (n, l(l+2))
    sinmcos=numpy.array(sinmcos).transpose()
    costhetav=costhetav[:, numpy.newaxis]
    sinthetav=sinthetav[:, numpy.newaxis]
    #debuge ~~~~
    #return legs,dlegs

    #build condition array A
    Ax = dlegs*(-sinthetav)*cossin
    Ay = ms*legs*sinmcos/sinthetav
    Az = -lplusones*legs*cossin
    
    #debugge
    #return Ax, Ay, Az
    #so far so good (maybe)

    #make big array
    A = numpy.concatenate((Ax, Ay, Az), axis=0)
    
    #nice mem
    del legs, dlegs, ms, lplusones, cossin, sinmcos, costhetav, sinthetav, Ax, Ay, Az
    
    At = A.transpose()

    g = numpy.matmul(numpy.matmul(numpy.linalg.inv(numpy.matmul(At,A)),At),data)

    gs = numpy.zeros((order+1,order+1))
    hs = gs.copy()

    for o, item in zip(orders, g):
        m,l = o
        if m < 0:
            hs[-m,l]=item
        else:
            gs[m,l]=item

    #it works!
    return gs,hs
    
