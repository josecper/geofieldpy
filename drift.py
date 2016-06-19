import xyzfield
from matplotlib import gridspec, pyplot
from mpl_toolkits.basemap import Basemap
import numpy, scipy
import scipy.signal, scipy.interpolate

#equidistant azimuthal, only used for curvature calculations
fake_map = Basemap(projection='npaeqd',boundinglat=0,lon_0=0,resolution='l')

def arc_dist(lat1, lon1, lat2, lon2, units="rad"):
    if units == "deg":
        lat1, lon1, lat2, lon2 = numpy.deg2rad((lat1, lon1, lat2, lon2))
    
    return numpy.arctan2(numpy.sqrt((numpy.cos(lat2)*numpy.sin(numpy.abs(lon2-lon1)))**2 + (numpy.cos(lat1)*numpy.sin(lat2) - numpy.sin(lat1)*numpy.cos(lat2)*numpy.cos(numpy.abs(lon2-lon1)))**2),numpy.sin(lat1)*numpy.sin(lat2) + numpy.cos(lat1)*numpy.cos(lat2)*numpy.cos(numpy.abs(lat2-lat1)))
    #return numpy.arccos(numpy.sin(lat1)*numpy.sin(lat2) + numpy.cos(lat1)*numpy.cos(lat2)*numpy.cos(lon2-lon1))
    
def avg_dist(lats, lons):
    
    dist = numpy.empty_like(lats)
    dist[0] = arc_dist(lats[0], lons[0], lats[1], lons[1])
    dist[-1] = arc_dist(lats[-1], lons[-1], lats[-2], lons[-2])
    dist[1:-1] = (arc_dist(lats[0:-2], lons[0:-2], lats[1:-1], lons[1:-1])+arc_dist(lats[1:-1], lons[1:-1], lats[2:], lons[2:]))/2
    return dist

def curvature(lats, lons, delta_time = 50, filtering=None):
    
    xx, yy = fake_map(slon, slat)
    difx, dify = numpy.gradient(xx)/6371000, numpy.gradient(yy)/6371000
    tan_len = numpy.sqrt(difx**2 + dify**2)/delta_time
    curv = (numpy.gradient(difx)**2 + numpy.gradient(dify)**2)/tan_len

    if filtering is None:
        return curv
    else:
        b, a = scipy.signal.butter(4, 1/filtering, btype="low")
        return scipy.signal.lfilter(b, a, curv)[int(filtering):]

def drift(lat, lon, gcomps, order=10):
    theta = numpy.deg2rad(90 - lat)
    phi = numpy.deg2rad(lon)

    declinations = numpy.empty_like(data.years)
    inclinations = numpy.empty_like(data.years)
    
    for i,y in enumerate(data.years):
        dec, inc, f, h = xyzfield.xyz2difh(*xyzfield.xyzfieldv2(gcomps[i,:],
                                                                phi, theta, order))
        declinations[i] = dec
        inclinations[i] = inc
        intensities[i] = f

    return declinations, inclinations, intensities

def rad2map(theta, phi):
    return numpy.rad2deg(numpy.pi - numpy.array(theta)), numpy.rad2deg(phi)

def map2rad(lat, lon):
    return numpy.deg2rad(90 - numpy.array(lat)), numpy.deg2rad(lon)
