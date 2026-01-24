import sys
from KDEpy import FFTKDE
import numpy as np


def progress(count, total, status=""):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    sys.stdout.write("[%s] %s%s ...%s\r" % (bar, percents, "%", status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)


def kde_1D(
    ax,
    data,
    cmap="jet",
    alpha=0.7,
    bw=0.25,
    color="b",
    fillcolor="lightblue",
    plotline=True,
):
    x, y = FFTKDE(kernel="gaussian", bw=bw).fit(data).evaluate()
    if plotline:
        ax.plot(x, y, color)
    ax.fill_between(x=x, y1=y, color=fillcolor)


def kde_joint(
    ax,
    data,
    cmap="jet",
    alpha=0.7,
    bw=0.25,
    N=32,
    maxz_scale=2,
    plotlines=False,
    n_grid_points=1024,
):
    # bw = (data.shape[0] * (2 + 2) / 4.)**(-1. / (2 + 4))
    grid_points = n_grid_points  # 2**10  # Grid points in each dimension
    # N = 8 # Number of contours
    kde = FFTKDE(bw=bw)
    grid, points = kde.fit(data).evaluate(grid_points)
    # The grid is of shape (obs, dims), points are of shape (obs, 1)
    x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = points.reshape(grid_points, grid_points).T
    # Plot the kernel density estimate
    maxz = z.max()
    minz = z.min()
    # Nlog = np.log10(np.logspace(minz,maxz,N))
    Nlog = np.geomspace(1e-15 * maxz, maxz_scale * maxz, 3 * N)
    # print("nlog", Nlog)
    if plotlines:
        ax.contour(x, y, z, Nlog, linewidths=0.5, colors="k", alpha=0.5)
    ax.contourf(x, y, z, Nlog, cmap=cmap, alpha=alpha)
