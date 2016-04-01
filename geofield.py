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

        timepairs=[(l,h,t) for l,h,t in zip(lowt,hight,times) if (l >= min(self.years) and h <= max(self.years))]
        validtimes=[p[2] for p in timepairs]

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

        return validtimes,numpy.array(diffs)

    def secularacceleration(self,times,interval=0.5,phiv=scipy.linspace(0.0,scipy.pi*2,100),thetav=scipy.linspace(0.01,scipy.pi-0.01,100),rparam=1.0):
        lowt=times-numpy.array(interval)
        hight=times+numpy.array(interval)

        timepairs=[(l,h,t) for l,h,t in zip(lowt,hight,times) if (l >= min(self.years) and h <= max(self.years))]
        validtimes=[p[2] for p in timepairs]

        g_high,h_high=self.interpolated([p[1] for p in timepairs])
        g_current,h_current=self.interpolated([p[2] for p in timepairs])
        g_low,h_low=self.interpolated([p[0] for p in timepairs])

        accs=[]

        for p,i in zip(timepairs,range(len(timepairs))):

            highx,highy,highz=xyzfieldv(g_high[i],h_high[i],phiv,thetav,rparam)
            lowx,lowy,lowz=xyzfieldv(g_low[i],h_low[i],phiv,thetav,rparam)
            currentx,currenty,currentz=xyzfieldv(g_current[i],h_current[i],phiv,thetav,rparam)

            accx=(lowx-2*currentx+highx)/((p[2]-p[0])*(p[1]-p[2]))
            accy=(lowy-2*currenty+highy)/((p[2]-p[0])*(p[1]-p[2]))
            accz=(lowz-2*currentz+highz)/((p[2]-p[0])*(p[1]-p[2]))

            accs.append((accx,accy,accz))

        return validtimes,numpy.array(accs)
    
def locationfield(lat,lon,x,y,z,phiv,thetav):

    theta=scipy.pi/2-numpy.deg2rad(lat)
    phi=numpy.deg2rad(lon)

    x_atlocation=scipy.interpolate.interp2d(thetav,phiv,x,kind="linear")(theta,phi)
    y_atlocation=scipy.interpolate.interp2d(thetav,phiv,y,kind="linear")(theta,phi)
    z_atlocation=scipy.interpolate.interp2d(thetav,phiv,z,kind="linear")(theta,phi)
    
    return numpy.array((x_atlocation,y_atlocation,z_atlocation))
