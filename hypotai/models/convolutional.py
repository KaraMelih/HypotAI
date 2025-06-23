import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorflow as tf
from tensorflow.keras import layers, models
from ..data import generate_triangle_data
from ..plotting import plot_triangle

def triangle_image_array(a, b, angle=90, size=(64, 64)):
    # Create figure and attach canvas
    fig = plt.figure(figsize=(2, 2), dpi=size[0] // 2)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    # Draw triangle
    plot_triangle(a, b, angle, ax=ax, annotation=False)
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 101)
    ax.set_aspect('equal')
    ax.axis("off")

    # Draw and get image buffer
    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    image = image.reshape((h, w, 4))  # RGBA

    # Convert to grayscale
    image = image[..., :3].mean(axis=2) / 255.0  # Normalize to [0,1]
    plt.close(fig)
    return image


def get_model(model_number:int = 1):
    """
    Returns a compiled Keras model based on the specified model number.
    Model 1 is a convolutional neural network for triangle regression.
    """
    if model_number == 1:
        return model1
    elif model_number == 2:
        return model2
    else:
        raise ValueError(f"Model {model_number} is not defined.")
    
    
# Define the model
model1 = models.Sequential([
    layers.Input(shape=(64, 64, 1)),  # Grayscale input

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output: regression value
])


# Define an alternative , each layer extracts more abstract features
## At each layer, each kernel learns more complex features 
model2 = models.Sequential([
    layers.Input(shape=(64, 64, 1)),  # Grayscale input

    layers.Conv2D(32, (5, 5), activation='relu'),  ### 32 filters, each 5x5 : learns broader features
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(8, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(8, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Output: regression value
])
