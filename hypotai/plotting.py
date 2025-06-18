import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots in older matplotlib
from sklearn.linear_model import LinearRegression
import math

def best_grid_shape(N) -> tuple:
    best = None
    min_diff = None
    for nrow in range(int(math.sqrt(N)), 0, -1):
        ncol = math.ceil(N / nrow)
        diff = abs(nrow - ncol)
        if (best is None) or (diff < min_diff):
            best = (nrow, ncol)
            min_diff = diff
    return best

def plot_regression_surface(model: LinearRegression, resolution=50, ax=None):
    """
    Plots the learned regression surface over a grid of (a, b).
    """
    # Create a meshgrid of (a, b)
    a_vals = np.linspace(0, 100, resolution)
    b_vals = np.linspace(0, 100, resolution)
    A, B = np.meshgrid(a_vals, b_vals)

    # Predict hypotenuse using the model
    X_grid = np.column_stack((A.ravel(), B.ravel()))
    C_pred = model.predict(X_grid).reshape(A.shape)

    # Compute ground truth (optional)
    C_true = np.sqrt(A**2 + B**2)

    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        
    # Plot model surface
    ax.plot_surface(A, B, C_pred, cmap="viridis", alpha=0.7, label="Model")

    # Optionally overlay ground truth
    ax.plot_wireframe(A, B, C_true, color="red", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Side a")
    ax.set_ylabel("Side b")
    ax.set_zlabel("Predicted Hypotenuse")
    ax.set_title("Linear Regression Surface vs. Ground Truth")
    plt.tight_layout()
    plt.show()

def center_text(ax, text, x, y, fontsize=12):
    """
    Centers text at a specific (x, y) position in the given axis.
    """
    # 1. Midpoint
    xm = (x[0] + x[1]) / 2
    ym = (y[0] + y[1]) / 2

    # 2. Direction vector
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # 3. Perpendicular (normal) vector, normalized
    length = np.hypot(dx, dy)
    nx = -dy / length
    ny = dx / length

    # 4. Shift midpoint by 1 unit along normal
    shift = 1
    xt = xm #+ nx * shift
    yt = ym #+ ny * shift

    # --- Compute angle in display coordinates ---
    # Transform the two points to display coordinates
    p1 = ax.transData.transform((x[0], y[0]))
    p2 = ax.transData.transform((x[1], y[1]))
    # Calculate angle in screen space
    angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))


    # 5. Angle in degrees for rotation
    # angle = np.degrees(np.arctan2(dy, dx))
    if angle > 90 or angle < -90:
        angle += 180
    ax.text(xt, yt, text, ha='center', rotation=angle, fontsize=fontsize, 
            bbox=dict(facecolor='none', alpha=0.5), rotation_mode='anchor')


def plot_triangle(a, b, angle=90, ax=None):
    """
    Plots a triangle given sides a and b, and the angle between them.
    If ax is provided, plots on that axis.
    """
    # Use law of cosines for non-right triangles
    if angle == 90:
        c = np.sqrt(a**2 + b**2)
    else:
        c = np.sqrt(a**2 + b**2 - 2*a*b*np.cos(np.radians(angle)))

    # Coordinates of the triangle vertices
    x = np.array([0, a, a - b * np.cos(np.radians(angle)), 0])
    y = np.array([0, 0, b * np.sin(np.radians(angle)), 0])

    if ax is None:
        plt.figure(figsize=(4, 4))
        ax = plt.gca()

    ax.plot(x, y, marker='o', color='blue', label='Triangle')
    ax.fill(x, y, alpha=0.3, color='blue')

    center_text(ax, f'a = {a:.1f}', x[:2], y[:2],)
    center_text(ax, f'b = {b:.1f}', x[[0,2]], y[[0,2]],)
    center_text(ax, f'c = {c:.1f}', x[[1,2]], y[[1,2]],)

    ax.set_xlabel("Side a")
    ax.set_ylabel("Side b")
    ax.grid()


def plot_triangles_in_a_grid(side1, side2, angles=90, gridsize=(5, 5), max_side=10):
    """
    Plots a grid of triangles with sides a and b.
    If more triangles are provided than the grid size,
    it will only plot the first `gridsize[0] * gridsize[1]` triangles.
    Each triangle is plotted with sides from side1 and side2.
    The grid size is specified as a tuple (nrow, ncol).
    """
    assert len(side1) == len(side2), "side1 and side2 must have the same length"
    if isinstance(angles, int):
        angles = [angles] * len(side1)
    else:
        assert len(angles) == len(side1), "angles must match the length of side1 and side2"
    nrow, ncol = gridsize if len(side1) > gridsize[0]*gridsize[1] else best_grid_shape(len(side1))
    fig, axs = plt.subplots(ncols=ncol, nrows=nrow, figsize=(15, 15))
    axs = np.atleast_2d(axs)  # Ensure axs is always 2D for consistency
    axs = axs.flatten()  # Flatten the array of axes for easy indexing
    
    for i in range(len(axs)):
        plot_triangle(side1[i], side2[i], angle=angles[i], ax=axs[i])
        axs[i].axis('off')  # Hide axes for individual plots
    plt.tight_layout()
    plt.show()
