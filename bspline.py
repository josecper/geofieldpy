import numpy, numpy.linalg
from matplotlib import pyplot

def bspline(t, h=2/3):

    t_abs = numpy.abs(t)

    y = numpy.zeros_like(t_abs)
    t_dom = t_abs[t_abs <= 2]
    y[t_abs <= 2] = h*(1 - 3/2*t_dom**2 + 3/4*t_dom**3)*(t_dom < 1) + (h/4*(2-t_dom)**3)*(t_dom >= 1)

    return y

def d2spline(t,h=2/3):

    pass

def condition_array(knot_points, times):

    #get knot distance
    width = (knot_points[-1]-knot_points[0])/(len(knot_points)-1)

    #grids
    tgrid, kgrid = numpy.meshgrid(times, knot_points, indexing="ij")
    
    a = bspline((tgrid-kgrid)/width)

    return a

def solve(times, data, knot_points, l=0, reg_degree=2):

    #get condition array
    n = len(knot_points)
    a = condition_array(knot_points, times)

    #create regularization array
    #d = -2*numpy.eye(n)+numpy.eye(n+1)[1:,:-1]+numpy.eye(n+1)[:-1,1:]
    #d = numpy.eye(n)-2*numpy.eye(n+1)[1:,:-1]+numpy.eye(n+2)[2:,:-2]
    d = numpy.diff(numpy.eye(n),n=reg_degree,axis=0)
    dd = d.transpose() @ d

    at = a.transpose()

    #adda = numpy.matmul(at,numpy.matmul(dd,at))
    #smash the it
    spline_coefs = (numpy.linalg.inv(at@a+l*dd) @ at) @ data

    #return pyplot.plot(times, numpy.matmul(a, spline_coefs), "k", times, data, "g+")
    #spline_coefs = numpy.matmul(numpy.matmul(numpy.linalg.inv(numpy.matmul(at, a)+l*adda),at),data)

    return spline_coefs

def spline_rep(times, spline_coefs, knot_points, summing=True):

    width = (knot_points[-1]-knot_points[0])/(len(knot_points)-1)
    tgrid, kgrid = numpy.meshgrid(times, knot_points, indexing="ij")

    y = bspline((tgrid-kgrid)/width)*spline_coefs

    
    return y.sum(axis=1)

def deboor_base(t,k,d):
    
    tgrid, kgrid = numpy.meshgrid(t,k,indexing="ij")
    kgridp1 = numpy.concatenate((kgrid[:,1:],kgrid[:,-1:]),axis=1)
    
    y=numpy.zeros_like(tgrid)
    
    if(d == 0):
        y[(tgrid >= kgrid) & (tgrid < kgridp1)] = 1
    else:
        kgrid_pd = numpy.concatenate((kgrid[:,d:],numpy.tile(kgrid[:,-1:],(1,d))),axis=1)
        kgrid_pdp1 = numpy.concatenate((kgrid_pd[:,1:],kgrid_pd[:,-1:]),axis=1)
        k_p1 = numpy.append(k[1:], k[-1])
        
        with numpy.errstate(divide="ignore", invalid="ignore"):
            dif_i=(tgrid-kgrid)/(kgrid_pd-kgrid)
            dif_im1=(kgrid_pdp1-tgrid)/(kgrid_pdp1-kgridp1)
        
        dif_i[numpy.isnan(dif_i)]=0
        dif_im1[numpy.isnan(dif_im1)]=0
        dif_i[numpy.isinf(dif_i)]=0
        dif_im1[numpy.isinf(dif_im1)]=0
        
        y = dif_i*deboor_base(t,k,d-1) + dif_im1*deboor_base(t,k_p1,d-1)
    
    #y[numpy.isnan(y)]=0
    return y
