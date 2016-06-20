import numpy
import sys
import datetime
from joblib import Parallel, delayed

def bootstrap(func, args, errors, error_type="gaussian", return_all=False, iterations=1000, *xargs, **kwargs):

    outputs = []
    
    for i in range(iterations):

        boot_args = [arg + error*numpy.random.randn(*arg.shape) for arg, error in zip(args,errors)]
        outputs.append(func(*boot_args, *xargs, **kwargs))

    outputs = numpy.array(outputs)
    average = outputs.sum(axis=0)/iterations
    stdev = numpy.sqrt(((outputs - average)**2).sum(axis=0)/iterations)

    if return_all:
        return average, stdev, outputs
    else:
        return average, stdev
    
    
def parallelize(func, args, errors, error_type="gaussian", return_all=False, n_jobs=4, iterations=100, debug=False, *xargs, **kwargs):
    
    outputs = []
    if debug:
        start = datetime.datetime.now()

    if (iterations % n_jobs) != 0:
        raise Exception("iterations must be a multiple of n_jobs")

    for i in range(iterations // n_jobs):

        if debug:
            sys.stdout.write("\r"+"{time} | iterations: {its} / {itsmax}".format(time=datetime.datetime.now()-start,
                                                                             its=i*n_jobs, itsmax = iterations))

        boot_args = [[arg + error*numpy.random.randn(*arg.shape) for arg, error in zip(args,errors)] for j in range(n_jobs)]

        outs = Parallel(n_jobs=n_jobs)(delayed(func)(*boot_arg, *xargs, **kwargs) for boot_arg in boot_args)
        outputs.extend(outs)

    outputs = numpy.array(outputs)
    average = outputs.sum(axis=0)/iterations
    stdev = numpy.sqrt(((outputs - average)**2).sum(axis=0)/iterations)

    if return_all:
        return average, stdev, outputs
    else:
        return average, stdev
