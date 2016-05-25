import numpy, numpy.linalg
import scipy, scipy.special, scipy.misc, scipy.sparse
import xyzfield, newleg
#debug
from matplotlib import pyplot

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
    g=numpy.zeros(order*(order+2)); g[0]=g0
    Ax,Ay,Az = condition_array_xyz(thetav,phiv,order)

    dec0, inc0, int0 = data
    dif0=numpy.concatenate((dec0,inc0,int0))
    dif=dif0.copy()

    # 2. calculate A arrays (dDIF vs. dg) (remember: constant part)
    for iteration in range(steps):
        x,y,z=xyzfield.xyzfieldv2(g,phiv,thetav, rparam=1.0, order=order) #small amount of points (supposedly)
        #debug
        #pyplot.show(xyzfield.xyzcontour(thetav,phiv,x,y,z,regular=False))
        
        dec,inc,intensity,horizontal=xyzfield.xyz2difh(x,y,z)
        
        Ad, Ai, Af = condition_array_dif(x,y,z,intensity,horizontal, Ax, Ay, Az, order)
        #remove emptys
        Ad=Ad[:,~numpy.isnan(dec0)]
        Ai=Ai[:,~numpy.isnan(inc0)]
        Af=Af[:,~numpy.isnan(int0)]

        dec=dec[~numpy.isnan(dec0)]
        inc=inc[~numpy.isnan(inc0)]
        intensity=intensity[~numpy.isnan(int0)]

        Adif=numpy.concatenate((Ad,Ai,Af), axis=1)

        old_dif=dif.copy()
        dif = numpy.concatenate((dec,inc,intensity))
        
        #get delta
        delta=dif0-dif
        #g = numpy.linalg.inv(Adif @ Adif.transpose()) @ Adif @ dif0
        g = g + numpy.linalg.inv(Adif @ Adif.transpose()) @ Adif @ delta
        print(g[:5], sum(abs(delta)))
    
    # 3. invert to obtain dg -> g
    # 4. synthesize XYZDIFH again
    # 5. goto 2
    #fix weird shit
    return g

def condition_array_xyz(thetav, phiv, order=13):

    mv,lv=newleg.degrees(order, start=1)
    leg,dleg=newleg.legendre(scipy.cos(thetav), order)

    cossin = numpy.zeros((len(phiv),order*(order+2)))
    cossin[:, mv >= 0] = numpy.cos(phiv[:, numpy.newaxis] @ abs(mv[numpy.newaxis, :]))[:, mv >= 0]
    cossin[:, mv < 0] = numpy.sin(phiv[:, numpy.newaxis] @ abs(mv[numpy.newaxis, :]))[:, mv < 0]

    sinmcos=numpy.zeros((len(phiv),order*(order+2)))
    sinmcos[:, mv >= 0] = numpy.sin(phiv[:, numpy.newaxis] @ abs(mv[numpy.newaxis, :]))[:, mv >= 0]
    sinmcos[:, mv < 0] = -numpy.cos(phiv[:, numpy.newaxis] @ abs(mv[numpy.newaxis, :]))[:, mv < 0]
    
    costhetav = numpy.cos(thetav)
    sinthetav = numpy.sin(thetav)

    Ax = cossin.transpose() * dleg * (-sinthetav)
    Ay = abs(mv[:, numpy.newaxis]) * leg * sinmcos.transpose() / sinthetav
    Az = -(lv + 1)[:, numpy.newaxis] * leg * cossin.transpose()

    return Ax, Ay, Az

def condition_array_dif(x, y, z, f, h, Ax, Ay, Az, order=13):

    #this is probably wrong :(
    Ad = (-y*Ax+x*Ay)/h**2
    Ai = (-x*z*Ax-y*z*Ay)/(h*f**2)+Az*h/f**2
    Af = (x*Ax+y*Ay+z*Az)/f
    
    return Ad, Ai, Af
