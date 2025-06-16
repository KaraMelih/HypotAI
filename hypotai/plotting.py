import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots in older matplotlib
from sklearn.linear_model import LinearRegression


def plot_regression_surface(model: LinearRegression, resolution=50):
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
