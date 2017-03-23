import numpy
import os

from matplotlib import pyplot
from mpl_toolkits.basemap import Basemap

PATH = "../data/rscha2d/mapas/"

files = [f for f in os.listdir(PATH) if f.endswith('.txt')]
theta_0p = numpy.deg2rad(26)
lat0 = numpy.rad2deg(theta_0p)

base = Basemap(projection="npaeqd", lon_0=0, boundinglat=90-lat0)

for fname in files:

    thetav, phiv, Dm, Im, Fm = numpy.loadtxt(PATH+fname).T

    print(len(thetav))

    latv = 90 - numpy.rad2deg(thetav)
    lonv = numpy.rad2deg(phiv)
    x_coord, y_coord = base(lonv, latv)

    fig, axes = pyplot.subplots(1, 3, subplot_kw={"aspect": "equal"})
    for ax, comp in zip(axes, (Dm, Im, Fm)):
        mappable = ax.tricontourf(x_coord, y_coord, comp, 60, cmap="Spectral")
        # ax.scatter(x_coord, y_coord, c=comp, cmap="Spectral")
        base.colorbar(mappable, ax=ax)
        base.drawcoastlines(ax=ax)
    fig.savefig(f"basura/{fname}.jpg")
    pyplot.close(fig)
