import numpy
import linecache

def load(filename):

    years = numpy.array([float(y) for y in linecache.getline(filename, 4).split()[3:-1]])
    everything = numpy.loadtxt(filename, comments="#", skiprows=4,
                               converters = {0 : lambda x: (-1 if x == b"h" else 1)}).T

    l, m = everything[1], everything[0]*everything[2]
    gcoefs = everything[3:-1]

    return years, (l, m), gcoefs
