import numpy, numpy.linalg
import scipy.interpolate
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

def solve(times, data, knot_points, l=0, reg_degree=2, spline_class="equal", degree=3, base=None, return_base = False):

    #get condition array
    n = len(knot_points)

    if base is None:
        if spline_class == "equal":
            a = condition_array(knot_points, times)
        elif spline_class == "deboor":
            a = deboor_base(knot_points, times, degree)
        else:
            raise Exception("it bad")
    else:
        a = base

    #create regularization array
    #d = -2*numpy.eye(n)+numpy.eye(n+1)[1:,:-1]+numpy.eye(n+1)[:-1,1:]
    #d = numpy.eye(n)-2*numpy.eye(n+1)[1:,:-1]+numpy.eye(n+2)[2:,:-2]
    d = numpy.diff(numpy.eye(n),n=reg_degree,axis=0)
    dd = d.transpose() @ d

    at = a.transpose()

    #adda = numpy.matmul(at,numpy.matmul(dd,at))
    spline_coefs = (numpy.linalg.inv(at @ a + l*dd) @ at) @ data

    #return pyplot.plot(times, numpy.matmul(a, spline_coefs), "k", times, data, "g+")
    #spline_coefs = numpy.matmul(numpy.matmul(numpy.linalg.inv(numpy.matmul(at, a)+l*adda),at),data)

    if return_base:
        return spline_coefs, a
    else:
        return spline_coefs

def spline_rep(times, spline_coefs, knot_points, summing=True):

    width = (knot_points[-1]-knot_points[0])/(len(knot_points)-1)
    tgrid, kgrid = numpy.meshgrid(times, knot_points, indexing="ij")

    y = bspline((tgrid-kgrid)/width)*spline_coefs

    
    return y.sum(axis=1)

def deboor_base(k,t,d):
    
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
        
        y = dif_i*deboor_base(k,t,d-1) + dif_im1*deboor_base(k_p1,t,d-1)

        #quick
        y[-1,:-d] = y[0, -(d+1)::-1]
    
    return y

def bootstrap_fit(times, data, time_error, data_error, times_out, n_knots=66, knot_points=None, l=260, iterations=1000, return_all=False):

    ndata = len(data)
    nout = len(times_out)

    if knot_points is None:
        #synthesize knot points
        t_min = min(times); t_max = max(times)
        k = numpy.array((t_min, t_min, t_min,
                         *tuple(numpy.linspace(t_min,t_max,n_knots-6)),
                         t_max, t_max, t_max))
    else:
        k = knot_points

    cs = numpy.zeros((iterations,nout))

    #synthesize original curve
    s0 = solve(times, data, k, l, spline_class="deboor")
    cs[0, :] = (deboor_base(k,times_out,3) * s0).sum(axis=1)


    for i in range(1,iterations):
        c_rand = numpy.random.randn(ndata)*data_error+data
        t_rand = numpy.random.randn(ndata)*time_error+times
        
        s = solve(t_rand, c_rand, k, l, spline_class="deboor")
        cs[i, :] = (deboor_base(k, times_out, 3) * s).sum(axis=1)

    c_avg = cs.sum(axis=0)/iterations
    stdev = numpy.sqrt(((cs-c_avg)**2).sum(axis=0)/iterations)

    if return_all:
        return c_avg, stdev, cs
    else:
        return c_avg, stdev

def bootstrap_fast(times, data, time_error, data_error, times_out, randomness=("normal","normal"), n_knots=66, knot_points=None, l=260, iterations=1000, clip=True, return_all=False):

    ndata=len(data)
    nout=len(times_out)

    random_funcs = {"normal" : numpy.random.randn,
                    "uniform" : lambda *args, **kwargs: 2*(numpy.random.rand(*args,**kwargs))-1}

    t_randf, c_randf = random_funcs[randomness[0]], random_funcs[randomness[1]]

    t_min = times[0]# - 3*time_error[0]
    t_max = times[-1]# + 3*time_error[-1]
    
    if knot_points is None:

        k = numpy.array((t_min, t_min, t_min,
                         *tuple(numpy.linspace(t_min,t_max,n_knots-6)),
                         t_max, t_max, t_max))
    else:
        k = knot_points

    #nice good yes
    cs = numpy.zeros((iterations,nout))
    
    s0, base = solve(times, data, k, l, spline_class="deboor", return_base=True)
    base_out = deboor_base(k, times_out, 3)

    cs[0,:] = (base_out*s0).sum(axis=1)

    #do the things
    #so fats

    c_rand = c_randf(iterations,ndata)*data_error+data
    if clip:
        t_rand = numpy.clip(t_randf(iterations,ndata)*time_error+times, t_min, t_max)
    else:
        t_rand = t_randf(iterations,ndata)*time_error+times
        
    for i in range(1,iterations):

        base_interpolated=scipy.interpolate.griddata(times_out, base_out, t_rand[i,:])
        s = solve(t_rand[i,:], c_rand[i,:], k, l, spline_class="deboor", base=base_interpolated)
        cs[i,:] = (base_out * s).sum(axis=1)

    c_avg = cs.sum(axis=0)/iterations
    stdev = numpy.sqrt(((cs-c_avg)**2).sum(axis=0)/iterations)

    if return_all:
        return c_avg, stdev, cs
    else:
        return c_avg, stdev

