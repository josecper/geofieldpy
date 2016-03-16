import numpy
import scipy

class SwarmData(object):

    def __init__(self,filename,sparse=True):
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
                        self.gcoefsdict=numpy.zeros((self,ntimes,self.nmax+1,self.nmax+1))
                        self.hcoefsdict=self.gcoefsdict.copy()
                    else:
                        elems=line.split()
                        l=int(elems[0])
                        m=int(elems[1])
                        values=elems[2:]
                        if m < 0:
                            for y,v in zip(self.years,values):
                                self.hcoefsdict[y][-m,l]=v
                        else:
                            for y,v in zip(self.years,values):
                                self.gcoefsdict[y][m,l]=v

    def interpolated(times):
        shape=(self.nmax+1,self.nmax+1)
        g=numpy.zeros((self.ntimes,*shape)) #ahora es 3D (t,m,l)
        h=g.copy() #hard copy

        for i,t in zip(range(len(times)),times):
            if t in self.gcoefsdict:
                pass

        
                        
    

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
