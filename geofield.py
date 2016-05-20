import numpy
from xyzfield import xyzfieldv #eliminar esta dependencia
import scipy, scipy.sparse, scipy.interpolate

class GaussCoefficientsData(object):

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

    def field_at_location(self, lat, lon, times=None, field="f", rparam=1.0):

        if times == None:
            times=numpy.array(self.years)

        thetav=numpy.array([scipy.pi/2-numpy.deg2rad(lat)])
        phiv=numpy.array([numpy.deg2rad(lon)])

        if field == "f":

            xyz = numpy.zeros((len(times),3))
            g,h=self.interpolated(times)

            for t,i in zip(times,range(len(times))):
                xyz[i,:]=numpy.array(xyzfieldv(g[i],h[i],phiv,thetav,rparam))
            return times, xyz[:,0], xyz[:,1], xyz[:,2]

        elif field == "s":

            vtimes,secular=self.secularvariation(times,interval=0.5,phiv=phiv,thetav=thetav)
            return vtimes, secular[:,0], secular[:,1], secular[:,2]

        elif field == "a":

            vtimes,acc=self.secularacceleration(times,interval=0.5,phiv=phiv,thetav=thetav)
            return vtimes, acc[:,0], acc[:,1], acc[:,2]

        else: raise Exception("you did a bad thing :(")

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

    def secular_power(self, times, interval=10.0/12.0, rparam=1.0):

        lowt=times-numpy.array(interval)
        hight=times+numpy.array(interval)

        timepairs=[(l,h,t) for l,h,t in zip(lowt,hight,times) if (l >= min(self.years) and h <= max(self.years))]
        validtimes=[p[2] for p in timepairs]

        g_high, h_high = self.interpolated([p[1] for p in timepairs])
        g_current, h_current = self.interpolated([p[2] for p in timepairs])
        g_low, h_low = self.interpolated([p[0] for p in timepairs])

        accs=[]

        for p,i in zip(timepairs, range(len(timepairs))):

            g_acc = ((g_low[i]-2*g_current[i]+g_high[i])/((p[2]-p[0])*(p[1]-p[2])))**2
            h_acc = ((h_low[i]-2*h_current[i]+h_high[i])/((p[2]-p[0])*(p[1]-p[2])))**2

            l_acc = (g_acc+h_acc).sum(axis=0)
            l_array = rparam**(2*numpy.arange(len(l_acc))+4)*(numpy.arange(len(l_acc))+1)

            accs.append(l_array*l_acc)

        return validtimes, numpy.array(accs)

            




class SwarmData(GaussCoefficientsData):

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

class ChaosData(GaussCoefficientsData):

    def __init__(self,filename,sparse=False):

        import scipy.sparse
        fs = open(filename, "r")

        self.years = [float(i) for i in fs.readline().split()]
        self.ntimes = len(self.years)
        self.nmin = 1; self.nmax = 20
        self.gcomp = scipy.zeros((self.ntimes,self.nmax*(self.nmax+2)+1))

        if sparse:
            self.g = scipy.sparse.csc_matrix(shape=(self.ntimes, self.nmax+1, self.nmax+1))
            self.h = self.g.copy()
        else:
            self.g = numpy.zeros((self.ntimes, self.nmax+1, self.nmax+1))
            self.h = self.g.copy()

        orders=[]
        for l in range(self.nmin,self.nmax+1):
            for m in range(0,l+1):
                if m == 0:
                    orders.append((l,m))
                else:
                    orders.append((l,m))
                    orders.append((l,-m))

        for i, order, line in zip(range(1,self.nmax+1),orders,fs):
            values=[float(v) for v in line.split()]
            self.gcomp[:,i] = numpy.array(values)
            l,m=order
            #print(l,m)
            if m < 0:
                self.h[:,-m,l] = numpy.array(values)
            else:
                self.g[:,m,l] = numpy.array(values)
            if l >= self.nmax:
                break



def locationfield(lat,lon,x,y,z,phiv,thetav):

    theta=scipy.pi/2-numpy.deg2rad(lat)
    phi=numpy.deg2rad(lon)

    x_atlocation=scipy.interpolate.interp2d(thetav,phiv,x,kind="linear")(theta,phi)
    y_atlocation=scipy.interpolate.interp2d(thetav,phiv,y,kind="linear")(theta,phi)
    z_atlocation=scipy.interpolate.interp2d(thetav,phiv,z,kind="linear")(theta,phi)

    return numpy.array((x_atlocation,y_atlocation,z_atlocation))
