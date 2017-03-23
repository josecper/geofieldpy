import numpy
import fibonacci_sphere
import trig

from matplotlib import pyplot
from mpl_toolkits.basemap import Basemap

thetav, phiv = fibonacci_sphere.grid(n=12000)
theta_0p = numpy.deg2rad(26)

in_cap = trig.angulardist(thetav, phiv, 0.0, 0.0) < theta_0p
thetav = thetav[in_cap]
phiv = phiv[in_cap]
print(len(thetav))

lat0 = numpy.rad2deg(theta_0p)
latv = 90 - numpy.rad2deg(thetav)
lonv = numpy.rad2deg(phiv)

base = Basemap(projection="npaeqd", lon_0=0, boundinglat=90-lat0)

x_coord, y_coord = base(lonv, latv)

for i in range(1, 60):
    z = numpy.sin(thetav/i) + numpy.cos(i*phiv)
    fig, axes = pyplot.subplots(1, 3, subplot_kw={"aspect": "equal"})
    for ax in axes:
        mappable = ax.tricontourf(x_coord, y_coord, z, 60, cmap="Spectral",
                                  vmin=-2, vmax=2)
        base.colorbar(mappable, ax=ax)
        base.drawcoastlines(ax=ax)
    fig.savefig(f"basura/test{i}.jpg")
