
import numpy as np
from matplotlib import pyplot as plt

def pcolormeshC(x_centers, y_centers, z, ax=None,
                            shading='auto', **pcolor_kwargs):
    """
    # Like pcolormesh, but takes 1D X and Y coordinates for centres of the pixels.
    
    Create a pcolormesh from a 2D array and 1D coordinate-center arrays.

    Parameters
    ----------
    x_centers : 1D array
        X coordinates of cell centers (length = number of columns in z)
    y_centers : 1D array
        Y coordinates of cell centers (length = number of rows in z)
    z : 2D array
        Data array with shape (len(y_centers), len(x_centers))
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on
    shading : str
        Passed to pcolormesh (default: 'auto')
    **pcolor_kwargs
        Extra kwargs passed to pcolormesh

    Returns
    -------
    pcm : QuadMesh
        The pcolormesh object
    """

    x_centers = np.asarray(x_centers)
    y_centers = np.asarray(y_centers)
    z = np.asarray(z)

    if z.shape != (len(y_centers), len(x_centers)):
        raise ValueError(
            f"z shape {z.shape} does not match "
            f"(len(y_centers), len(x_centers)) = "
            f"({len(y_centers)}, {len(x_centers)})"
        )

    # Convert centers -> edges
    def centers_to_edges(c):
        dc = np.diff(c)

        edges = np.empty(len(c) + 1)

        # Interior edges
        edges[1:-1] = c[:-1] + dc / 2

        # Extrapolate outer edges
        edges[0] = c[0] - dc[0] / 2
        edges[-1] = c[-1] + dc[-1] / 2

        return edges

    x_edges = centers_to_edges(x_centers)
    y_edges = centers_to_edges(y_centers)

    if ax is None:
        fig, ax = plt.subplots()

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        z,
        shading=shading,
        **pcolor_kwargs
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return pcm