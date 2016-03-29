import numpy
from xyzfield import xyzfieldv #eliminar esta dependencia
import scipy, scipy.sparse, scipy.interpolate

class SwarmData(object):

    def __init__(self,filename,sparse=False):
            import scipy.sparse
            fs=open(filename,"r")
            yearsread=False
            for line in fs:
                if not line.startswith("#"):
                    if not yearsread:
                        self.nmin,self.nmax,self.ntimes,self.spline_order,self.nstep=(int(x) for x in line.split()[:5])
                        line=fs.readline()
                        self.years=[float(i) for i in line.split()]
                        yearsread=True;
                        #init arrays
                        if(sparse):
                            self.g=scipy.sparse.csc_matrix(shape=(self.ntimes,self.nmax+1,self.nmax+1))
                            self.h=self.g.copy()
                        else:
                            self.g=numpy.zeros((self.ntimes,self.nmax+1,self.nmax+1))
                            self.h=self.g.copy()
                    else:
                        elems=line.split()
                        l=int(elems[0])
                        m=int(elems[1])
                        values=elems[2:]
                        if m < 0:
                            for y,v,i in zip(self.years,values,range(self.ntimes)):
                                self.h[i,-m,l]=float(v)
                        else:
                            for y,v,i in zip(self.years,values,range(self.ntimes)):
                                self.g[i,m,l]=float(v)

    def interpolated(self,times):
        shape=(self.nmax+1,self.nmax+1)
        gint=numpy.zeros((len(times),*shape)) #ahora es 3D (t,m,l)
        hint=gint.copy() #hard copy
        t=numpy.array(self.years)

        for m in range(self.nmax+1):
            for l in range(self.nmax+1):
                gy=self.g[:,m,l]
                hy=self.h[:,m,l]               
                ginterpolant=scipy.interpolate.InterpolatedUnivariateSpline(t,gy,k=5)
                hinterpolant=scipy.interpolate.InterpolatedUnivariateSpline(t,hy,k=5)

                gint[:,m,l]=ginterpolant(numpy.array(times))
                hint[:,m,l]=hinterpolant(numpy.array(times))
                
        return gint,hint

    def secularvariation(self,times,interval=0.5,phiv=scipy.linspace(0.0,scipy.pi*2,100),thetav=scipy.linspace(0.01,scipy.pi-0.01,100),rparam=1.0):
        lowt=times-numpy.array(interval)
        hight=times+numpy.array(interval)

        timepairs=[(l,h) for l,h in zip(lowt,hight) if (l >= min(self.years) and h <= max(self.years))]

        g_high,h_high=self.interpolated([p[1] for p in timepairs])
        g_low,h_low=self.interpolated([p[0] for p in timepairs])

        diffs=[]

        for p,i in zip(timepairs,range(len(timepairs))):

            highx,highy,highz=xyzfieldv(g_high[i],h_high[i],phiv,thetav,rparam)
            lowx,lowy,lowz=xyzfieldv(g_low[i],h_low[i],phiv,thetav,rparam)

            diffx=(highx-lowx)/(p[1]-p[0])
            diffy=(highy-lowy)/(p[1]-p[0])
            diffz=(highz-lowz)/(p[1]-p[0])

            diffs.append((diffx,diffy,diffz))

        return numpy.array(diffs)
        
def locationfield(lat,lon,x,y,z,phiv,thetav):

    theta=scipy.pi/2-numpy.deg2rad(lat)
    phi=numpy.deg2rad(lon)

    x_atlocation=scipy.interpolate.inqterp2d(phiv,thetav,x,kind="linear")(phi,theta)
    y_atlocation=scipy.interpolate.interp2d(phiv,thetav,y,kind="linear")(phi,theta)
    z_atlocation=scipy.interpolate.interp2d(phiv,thetav,z,kind="linear")(phi,theta)
    
    return numpy.array((x_atlocation,y_atlocation,z_atlocation))

    
def interpolatecoefs(times,gcoefsd,hcoefsd):
    """
    gcoefsd = dictionary of { float : m*l float64 array }
    hcoefsd = same
    times = array of years

    returns g,h : t.shape*m*l
    """

    s=(next(iter(gcoefsd.values()))).shape
    g=numpy.zeros((len(times),s[0],s[1]))
    h=g.copy()
    
    for i,t in zip(range(len(times)),times):
        if t in gcoefsd:
            g[i,:,:]=gcoefsd[t].toarray()
            h[i,:,:]=hcoefsd[t].toarray()
        else:
            prev_time=max([y for y in gcoefsd.keys() if y < t])
            prev_g=gcoefsd[prev_time]
            prev_h=hcoefsd[prev_time]

            next_time=min([y for y in gcoefsd.keys() if y > t])
            next_g=gcoefsd[next_time]
            next_h=hcoefsd[next_time]

            interpolant=(t-prev_time)/(next_time-prev_time)

            g[i,:,:]=(interpolant*next_g+(1-interpolant)*prev_g).toarray()
            h[i,:,:]=(interpolant*next_h+(1-interpolant)*prev_h).toarray()

    return g,h
