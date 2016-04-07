import scipy
import scipy.special
import scipy.misc
import numpy
import sys


    #esto es un poco el mal

    
def umap(callable, thing):
    if(hasattr(thing, "__iter__")):
        return list(map(callable, thing))
    else:
        return [callable(thing)]

def load(filename):
    fs=open(filename, "r")
    gcoefs=numpy.zeros((19, 19))
    hcoefs=numpy.zeros((19, 19))
    
    for line in fs.readlines()[5:]:
        elems=line.split()
        l=int(elems[0])
        m=int(elems[1])
        value=float(elems[2])  
        if(m < 0):
            hcoefs[-m,l]=value
        else:
            gcoefs[m,l]=value
            
    return(gcoefs,hcoefs)

def loadyear(filename, years=(0,), sparse=True):
    """
    returns (g,h), dictionaries of arrays of Gauss coefficients extracted from file filename,
    and indexed by year. for the columns specified in years as a tuple of integers.

    if the sparse argument is True, then g and h are scipy.sparse.csr_matrix
    sparse arrays.
    """
    import scipy.sparse
    fs=open(filename,"r")
    yearsread=False
    for line in fs:
        if not line.startswith("#"):
            if not yearsread:
                #hacccck (saltarse una línea con numeros misteriosos)
                nmin,nmax,ntimes,spline_order,nstep=line.split()[:5]
                line=fs.readline()
                yeardates=[list(map(float,line.split()))[i] for i in years]
                yearsread=True
                #inicializar arrays
                gcoefsdict={y : numpy.zeros((19,19)) for y in yeardates}
                hcoefsdict={y : numpy.zeros((19,19)) for y in yeardates}
            else:
                elems=line.split()
                l=int(elems[0])
                m=int(elems[1])
                values=[float(elems[i+2]) for i in years]
                if(m < 0):
                    for y,v in zip(yeardates,values):
                        hcoefsdict[y][-m,l]=v
                else:
                    for y,v in zip(yeardates,values):
                        gcoefsdict[y][m,l]=v

    if sparse:
        #convertir a sparse arrays para ahorrar memoria
        gcoefsdictsp={y: scipy.sparse.csr_matrix(gcoefsdict[y]) for y in gcoefsdict}
        hcoefsdictsp={y: scipy.sparse.csr_matrix(hcoefsdict[y]) for y in hcoefsdict}
        return(yeardates,gcoefsdictsp, hcoefsdictsp)
    else:
        return(yeardates,gcoefsdict,hcoefsdict)
    
    
def precalc(theta,order=13):
    legendre,dlegendre=scipy.special.lpmn(order+1,order+1,scipy.cos(theta))
    return(legendre,dlegendre)

def xyzfield(gcoefs,hcoefs,phi,theta,rparam=1.0,order=13):
    #no usar esta función, es peor en rendimiento
    x,y,z=0,0,0


    legendre,dlegendre=scipy.special.lpmn(order+1,order+1,scipy.cos(theta))
    
    for l in range(1,order+1):
        for m in range(0,l+1):
            deltax=rparam**(l+2)*(gcoefs[m,l]*scipy.cos(m*phi)+hcoefs[m,l]*scipy.sin(m*phi))*dlegendre[m,l]*(-scipy.sin(theta))
            deltay=rparam**(l+2)*(gcoefs[m,l]*scipy.sin(m*phi)-hcoefs[m,l]*scipy.cos(m*phi))*m*legendre[m,l]/(scipy.sin(theta))
            deltaz=rparam**(l+2)*(l+1)*(gcoefs[m,l]*scipy.cos(m*phi)+hcoefs[m,l]*scipy.sin(m*phi))*legendre[m,l]

            x+=deltax
            y+=deltay
            z+=deltaz
            
    return(x,y,z)

def xyzfieldv(gcoefs,hcoefs,phiv,thetav,rparam=1.0,order=13):
    """
    returns (x,y,z), a tuple of arrays representing the corresponding components of
    the geomagnetic field over a grid of points given by phiv and thetav, calculated
    using coefficient arrays gcoefs and hcoefs up to l=order, with distance to the center
    of the Earth given by rparam=r/a.

    if phiv and thetav are scalars, (x,y,z) will be a tuple of scalars instead.
    """
    #función pseudo-universal sobre phiv,thetav gracias a la magia negra
    thetagrid,phigrid=numpy.meshgrid(thetav,phiv,indexing="xy")

    x=numpy.zeros_like(thetagrid)
    y=numpy.zeros_like(thetagrid)
    z=numpy.zeros_like(thetagrid)
        
    #matriz de normalización de schmidt
    lgrid,mgrid=scipy.meshgrid(numpy.arange(0,order+1),numpy.arange(0,order+1),indexing="xy")
    schmidt=(-1)**mgrid*scipy.sqrt((2-numpy.equal(mgrid,0))*scipy.misc.factorial(lgrid-mgrid)/scipy.misc.factorial(lgrid+mgrid))
    #precalcular las funciones de legendre
    legendre=numpy.array(umap(lambda x:numpy.multiply(scipy.special.lpmn(order,order,x),(schmidt,schmidt)), scipy.cos(thetav)))
    #polinomios
    plegendre=legendre[:,0,:,:]
    #derivadas
    dlegendre=legendre[:,1,:,:]
    
    for l in range(1,order+1):
        rparamexp=rparam**(l+2)
        for m in range(0,l+1):
            x+=rparamexp*(gcoefs[m,l]*scipy.cos(m*phigrid)+hcoefs[m,l]*scipy.sin(m*phigrid))*dlegendre[:,m,l]*(-scipy.sin(thetagrid))
            y+=rparamexp*(gcoefs[m,l]*scipy.sin(m*phigrid)-hcoefs[m,l]*scipy.cos(m*phigrid))*m*plegendre[:,m,l]/(scipy.sin(thetagrid))
            z-=rparamexp*(l+1)*(gcoefs[m,l]*scipy.cos(m*phigrid)+hcoefs[m,l]*scipy.sin(m*phigrid))*plegendre[:,m,l]

    if(x.shape == (1,1)):
        return (x.flat[0], y.flat[0], z.flat[0])
    else:
        return (x,y,z)

def xyz2dif(x,y,z,units="radians"):

    xysquared=x**2+y**2

    declination=numpy.arctan2(y,x)
    inclination=numpy.arctan2(z,numpy.sqrt(xysquared))

    if units=="degrees":
        declination=numpy.rad2deg(declination)
        inclination=numpy.rad2deg(inclination)
    
    intensity=numpy.sqrt(xysquared+z**2)

    return declination,inclination,intensity
    
def xyztime(gcoefsdict,hcoefsdict,phi,theta,t,rparam=1.0,order=13):
    """
    returns the x,y,z components of the field at coordinates or coordinate vectors phi and theta, at time t
    as given by measured or linearly interpolated data as appropiate
    """
    return xyzfieldv(*ghinterp(gcoefsdict, hcoefsdict, t), phi, theta, rparam, order)

    
def ghinterp(gcoefsdict,hcoefsdict,t):
    """
    returns (g,h) arrays of Gauss coefficients at time (in years) t, as given by the
    dictionaries of arrays g and h and the iterable of measurement dates yeardates. if t is not
    a measurement date, a linear interpolation between the two closest dates will be returned instead.

    this function preserves the density of the coefficient arrays (i. e. will return sparse arrays
    if the original arrays are sparse).
    """
    yeardates=list(gcoefsdict.keys())
    
    if(t in yeardates):
        #this saves like half a cpu cycle probably
        return(gcoefsdict[t], hcoefsdict[t])
    
    lowtime=max([y for y in yeardates if y < t])
    hightime=min([y for y in yeardates if y > t])

    g=(gcoefsdict[lowtime]*(hightime-t)+gcoefsdict[hightime]*(t-lowtime))/(hightime-lowtime)
    h=(hcoefsdict[lowtime]*(hightime-t)+hcoefsdict[hightime]*(t-lowtime))/(hightime-lowtime)
    return (g,h)


def xyzcontour(theta,phi,x,y,z,vmin=None,vmax=None,cmap="bwr",projection="robin",mode="xyz",units="nT",time=None,string="{0}"):

    from matplotlib import pyplot, colors
    from mpl_toolkits.basemap import Basemap

    if projection=="cyl":
        base=Basemap(projection="cyl",
                     llcrnrlat=90-scipy.rad2deg(max(theta)),
                     urcrnrlat=90-scipy.rad2deg(min(theta)),
                     llcrnrlon=scipy.rad2deg(min(phi)),
                     urcrnrlon=scipy.rad2deg(max(phi)),
                     #llcrnrlat=-90, urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,
                     resolution="l")
    elif projection=="robin":
        base=Basemap(projection="robin",
                     lon_0=0.0)
    else: raise Exception("bad projection :'(")

    xtrans=numpy.rad2deg(phi)
    ytrans=90-numpy.rad2deg(theta)

    fig=pyplot.figure(figsize=(10,13))
    axis1=fig.add_subplot(311);axis1.set_title(string.format("X",str(time)))
    axis2=fig.add_subplot(312);axis2.set_title(string.format("Y",str(time)))
    axis3=fig.add_subplot(313);axis3.set_title(string.format("Z",str(time)))

    for a in (axis1,axis2,axis3):
        base.drawcoastlines(ax=a)
        base.drawparallels(numpy.arange(-60.,90.,30.),ax=a)
        base.drawmeridians(numpy.arange(0.,420.,60.),labels=[0,0,0,1],fontsize=10,ax=a)
        base.drawmapboundary(ax=a)

    if mode == "dif":
        xycmap=colors.LinearSegmentedColormap(
            "crisisperrotini",
            segmentdata={
                "red":[(0.0, 0.0, 0.0),
                       (0.5, 1.0, 1.0),
                       (0.75,1.0, 1.0),
                       (1.0, 0.0, 0.0)],
                "green":[(0.0, 0.0, 0.0),
                         (0.5, 1.0, 1.0),
                         (1.0, 0.0, 0.0)],
                "blue":[(0.0, 0.0, 0.0),
                        (0.25, 1.0, 1.0),
                        (0.5, 1.0, 1.0),
                        (1.0, 0.0, 0.0)]
            })
        zcmap=cmap
    else:
        xycmap=zcmap=cmap

    xx,yy=numpy.meshgrid(xtrans,ytrans)
    
    if (not vmin or not vmax):
        xmax=numpy.max(numpy.abs(x))
        ymax=numpy.max(numpy.abs(y))
        zmax=numpy.max(numpy.abs(z))          
        m=base.contourf(xx,yy,x.transpose(),31,latlon=True,ax=axis1,vmin=-xmax,vmax=xmax,cmap=xycmap)
        cbar=base.colorbar(mappable=m,ax=axis1)
        cbar.set_label(units)
        m=base.contourf(xx,yy,y.transpose(),31,latlon=True,ax=axis2,vmin=-ymax,vmax=ymax,cmap=xycmap)
        cbar=base.colorbar(mappable=m,ax=axis2)
        cbar.set_label(units)        
        m=base.contourf(xx,yy,z.transpose(),31,latlon=True,ax=axis3,vmin=-zmax,vmax=zmax,cmap=zcmap);
        cbar=base.colorbar(mappable=m,ax=axis3)
        cbar.set_label(units)
    else:
        base.contourf(xx,yy,x.transpose(),31,latlon=True,ax=axis1,vmin=vmin,vmax=vmax,cmap=xycmap)
        base.contourf(xx,yy,y.transpose(),31,latlon=True,ax=axis2,vmin=vmin,vmax=vmax,cmap=xycmap)
        base.contourf(xx,yy,z.transpose(),31,latlon=True,ax=axis3,vmin=vmin,vmax=vmax,cmap=zcmap)
        
    return fig
                 

def xyzplot(theta,phi,x,y,z,vmin=-70000, vmax=70000,cmap="bwr"):
    """
    plots the x,y, and z components of the magnetic field in three separate color plots, over
    a world map in cylindrical equidistant projection, given point coordinate vectors theta and phi
    (work in other projections is in progress).    
    """    
    from matplotlib import pyplot
    from mpl_toolkits.basemap import Basemap

    base=Basemap(projection="cyl",
                 llcrnrlat=90-scipy.rad2deg(max(theta)),
                 urcrnrlat=90-scipy.rad2deg(min(theta)),
                 llcrnrlon=scipy.rad2deg(min(phi)),
                 urcrnrlon=scipy.rad2deg(max(phi)),
                 #llcrnrlat=-90, urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,
                 resolution="l")
    
    #xtrans=numpy.rad2deg(phi)-180
    #ytrans=numpy.rad2deg(theta)-90

    #xtrans=numpy.rad2deg(phi)
    #ytrans=90-numpy.rad2deg(theta)

    ytrans=90-numpy.rad2deg(theta)
    xtrans=numpy.rad2deg(phi)
    fig=pyplot.figure(figsize=(10,13))
    axis1=fig.add_subplot(311);axis1.set_title("X")
    axis2=fig.add_subplot(312);axis2.set_title("Y")
    axis3=fig.add_subplot(313);axis3.set_title("Z")

    base.drawcoastlines(ax=axis1)
    base.drawcoastlines(ax=axis2)
    base.drawcoastlines(ax=axis3)

    pyplot.set_cmap(cmap)
    #axis1.pcolor(xtrans,ytrans,numpy.rot90(x),vmin=vmin,vmax=vmax)
    #axis2.pcolor(xtrans,ytrans,numpy.rot90(y),vmin=vmin,vmax=vmax)
    #cplot=axis3.pcolor(xtrans,ytrans,numpy.rot90(z),vmin=vmin,vmax=vmax)

    axis1.pcolormesh(xtrans,ytrans,x.transpose(),vmin=vmin,vmax=vmax,shading="gouraud")
    axis2.pcolormesh(xtrans,ytrans,y.transpose(),vmin=vmin,vmax=vmax,shading="gouraud")
    cplot=axis3.pcolormesh(xtrans,ytrans,z.transpose(),vmin=vmin,vmax=vmax,shading="gouraud")
    
    pyplot.colorbar(mappable=cplot,orientation="vertical",ax=[axis1,axis2,axis3],aspect=40,format="%1.0f nT")
#   fig.show()
    return fig

def plotfield(filename="MCO_2C.DBL", resolution=(200,200), rparam=1.0, order=13, projection="cyl", vmin=-70000, vmax=+70000):
    """
    plot the magnetic field
    """
    from matplotlib import pyplot
    from mpl_toolkits.basemap import Basemap

    g,h=load(filename)

    theta=scipy.linspace(0.01,scipy.pi-0.01,resolution[1])
    phi=scipy.linspace(0.0,2*scipy.pi,resolution[0])

    x,y,z=xyzfieldv(g,h,phi,theta,rparam,order)

    return xyzplot(theta,phi,x,y,z,vmin,vmax)
