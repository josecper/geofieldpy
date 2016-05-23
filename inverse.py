import numpy, numpy.linalg
import scipy, scipy.special, scipy.misc, scipy.sparse
import xyzfield, newleg

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
    
def invert_dif(thetav, phiv, data, order=13, g0=-30000, steps=5):
    # 0. initial conditions (g)
    # 1. calculate XYZDIFH from that
    g=numpy.zeros(order*(order+2)+1); g[1]=g0
    
    x,y,z=xyzfield.xyzfieldv2(g,phiv,thetav) #small amount of points (supposedly)
    dec,inc,intensity,horizontal=xyzfield.xyz2difh(x,y,z)

    # 2. calculate A arrays (dDIF vs. dg) (remember: constant part)
    
    # 3. invert to obtain dg -> g
    # 4. synthesize XYZDIFH again
    # 5. goto 2
    
    pass

def condition_array_xyz(thetav, phiv, order=13):

    mv,lv=newleg.degrees(order)
    leg,dleg=newleg.legendre(scipy.cos(thetav), order)

    cossin = numpy.zeros((len(phiv),order*(order+2)+1))
    #cossin[mv >= 0] = 
    sinmcos=numpy.zeros((len(phiv),order*(order+2)+1))
    
