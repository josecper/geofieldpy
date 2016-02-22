import numpy
from scipy import randn, spatial
from matplotlib import pyplot

points=randn(5,2)
tri=spatial.Delaunay(points, incremental=True, qhull_options="Qt Qi Q6 Q5")

def onclick(event):
    if(event.inaxes):
        new_point=numpy.array([[event.xdata,event.ydata]])
        tri.add_points(new_point)
        event.inaxes.cla()
        event.inaxes.triplot(tri.points[:,0],tri.points[:,1], tri.simplices, color="k")
#       event.inaxes.tripcolor(tri.points[:,0], tri.points[:,1], tri.simplices, cmap=pyplot.cm.rainbow)
        ax.axis([-3,3,-3,3])
        event.canvas.draw()

fig=pyplot.figure()
ax=fig.add_subplot(111)
ax.triplot(tri.points[:,0],tri.points[:,1], tri.simplices, color="k")
ax.axis([-3,3,-3,3])
fig.show()

connection=fig.canvas.mpl_connect('button_press_event', onclick)
