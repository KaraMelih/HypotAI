# hypotai/data.py

import numpy as np
import pandas as pd
from typing import Literal


def generate_triangle_data(
    n_samples: int = 100_000,
    angle_mode: Literal["right", "random"] = "right",
    seed: int = 42) -> pd.DataFrame:
    """
    Generates triangle side lengths and computes hypotenuse or third side.

    Parameters:
    - n_samples: Number of triangles to generate.
    - angle_mode: "right" for 90°, "random" for a random angle between (1°, 179°)
    - seed: Random seed for reproducibility.

    Returns:
    - pd.DataFrame with columns: a, b, angle_deg, c

    # this is a test change
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.1, 100.0, n_samples)
    b = rng.uniform(0.1, 100.0, n_samples)

    if angle_mode == "right":
        angle_deg = np.full(n_samples, 90.0)
        c = np.sqrt(a**2 + b**2)
    elif angle_mode == "random":
        angle_deg = rng.uniform(1, 179, n_samples)
        angle_rad = np.deg2rad(angle_deg)
        c = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(angle_rad))
    else:
        raise ValueError("angle_mode must be 'right' or 'random'")

    return pd.DataFrame({"a": a, "b": b, "angle_deg": angle_deg, "c": c})
