from mpl_toolkits.basemap import Basemap
import matplotlib
from matplotlib import pyplot
import numpy
nicebwr = matplotlib.colors.LinearSegmentedColormap.from_list("nicebwr",["#094da0","#FFFFFF","#ef1a2d"])

import scha

def cute_lines(base_trans, ax):
    
    base_trans.drawmeridians(numpy.arange(0, 360, 60), latmax=90, ax=ax, color="black", linewidth=0.4)
    base_trans.drawparallels(numpy.linspace(0, 90, 15), ax=ax, color="black", linewidth=0.4)
    base_trans.drawcoastlines(ax=ax, linewidth=0.3)

def component_residual_plot(thetav, phiv, theta_0, theta_c, phi_c, components, residuals=None, dots=False, scales=None, cmaps=None, titles=None, lims=None, cbar=True, **pt_args):

    lon_c, lat_c = numpy.rad2deg((phi_c, numpy.pi/2 - theta_c))

    if scales is None:
        scales = ("symmetric", "symmetric", "symmetric")

    if cmaps is None:
        cmaps = (nicebwr, nicebwr, nicebwr)

    if titles is None:
        titles = ("", "", "")

    if lims is None:
        lims = (None, None, None)
    else:
        scales = ("custom", "custom", "custom")

    base = Basemap(projection="npaeqd", lon_0 = 0, boundinglat=90-numpy.rad2deg(theta_0))
    print(lon_c, lat_c)
    base2 = Basemap(projection="aeqd", lon_0 = lon_c, lat_0 = lat_c, lat_ts=45.0,
                    width=base.xmax, height=base.ymax, resolution="l")

    if residuals is not None:
        fig, axes = pyplot.subplots(2, 3, figsize=(9, 5), subplot_kw={'aspect': 'equal'})
        res_axes = axes[0]
        field_axes = axes[1]

        for res, ax in zip(residuals, res_axes):
            base.colorbar(scha.polar_tricontour(res, thetav, phiv, theta_0, ax=ax, base=base, cmap="PRGn", **pt_args),
                          ax=ax, fig=fig, location="right")
            cute_lines(base2, ax)
            if dots: base.scatter(numpy.rad2deg(phiv),90-numpy.rad2deg(thetav), s=1, marker=".", color="black",latlon=True,ax=ax)
    else:
        fig, field_axes = pyplot.subplots(1, 3, figsize=(18,6),
                                          #subplot_kw={'aspect':'equal'}
        )

    for comp, ax, scale, cmap, lim, title in zip(components, field_axes, scales, cmaps, lims, titles):
        if cbar:
            base.colorbar(scha.polar_tricontour(comp, thetav, phiv, theta_0, ax=ax,
                                            base=base, cmap=cmap, scale=scale, lims=lim, **pt_args),
                          ax=ax, fig=fig, location='right', format='%.1f')
        else:
            scha.polar_tricontour(comp, thetav, phiv, theta_0, ax=ax,
                                  base=base, cmap=cmap, scale=scale, lims=lim, **pt_args)
        #tric=scha.polar_tricontour(comp, thetav, phiv, theta_0, ax=ax, base=base, cmap=cmap, scale=scale, **pt_args)
        #cbar=fig.colorbar(tric, ax=ax, orientation="vertical", shrink=0.7)
        ax.set_title(title)
        cute_lines(base2, ax)
        #cbar.spines['bottom']._adjust_location()
        if dots: base.scatter(numpy.rad2deg(phiv),90-numpy.rad2deg(thetav), s=1,  marker=".", color="black",latlon=True,ax=ax)

    #fig.tight_layout()
    #pyplot.show(fig)
    return fig

def super_video_frame(thetav, phiv, theta_0, theta_c, phi_c, components, base_cap, base_world, fname, dots=False, scales=None, cmaps=None, **pt_args):

    lon_c, lat_c = numpy.rad2deg((phi_c, numpy.pi/2 - theta_c))

    if scales is None:
        scales = ("symmetric", "symmetric", "symmetric")

    if cmaps is None:
        cmaps = (nicebwr, nicebwr, nicebwr)

    lonv = numpy.rad2deg(phiv)
    latv = 90 - numpy.rad2deg(thetav)
    x_coord, y_coord = base_cap(lonv, latv)

    fig, field_axes = pyplot.subplots(1, 3, figsize=(15, 5), subplot_kw = {'aspect' : 'equal'})

    for comp, ax, scale, cmap in zip(components, field_axes, scales, cmaps):
        tric = scha.polar_tricontour(comp, thetav, phiv, theta_0, ax=ax,
                                     base=base_cap, cmap=cmap, scale=scale, **pt_args)

        tric = ax.tricontourf(x_coord, y_coord, comp, 60, cmap="Spectral")
        cbar = base.colorbar(tric, ax=ax)
        
        cute_lines(base_world, ax)
        #ax.spines['left']._adjust_location()
        if dots: base_cap.scatter(numpy.rad2deg(phiv),90-numpy.rad2deg(thetav), s=1,
                                  marker=".", color="black",latlon=True,ax=ax)

    fig.savefig(fname)
    pyplot.close(fig)
