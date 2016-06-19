import numpy

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
