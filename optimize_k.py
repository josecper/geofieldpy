import numpy

def kparam(w, decs, incs):
    n = sum(w)
    xu = numpy.sin(decs)*numpy.cos(incs)
    yu = numpy.cos(decs)*numpy.cos(incs)
    zu = numpy.sin(incs)
    rsum = numpy.sqrt((xu*w).sum()**2 + 
                      (yu*w).sum()**2 +
                      (zu*w).sum()**2)
    return (n - 1)/(n - rsum) 


def optimize_k(decs, incs, nmin=2):
    ntotal = len(decs)
    wranges = tuple([0, 1] for i in range(ntotal))
    combs = numpy.array(
        numpy.meshgrid(*wranges)).T.reshape(-1, ntotal)
    w = numpy.ones_like(decs)
    maxk = kparam(w, decs, incs)
    maxw = w.copy()
    for w in combs:
        if(sum(w) < nmin):
            continue
        k = kparam(w, decs, incs)
        if k > maxk:
            maxk = k
            maxw = w

    return maxw, maxk
