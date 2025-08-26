from __future__ import annotations

from typing import Callable, Dict, Tuple
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate as shp_translate
from rasterio.features import rasterize
from affine import Affine
import matplotlib.pyplot as plt
import pandas as pd

from utils import _makeCheckerboard

# ------------------------------
# Typing & Aggregates
# ------------------------------

AggregateFunc = Callable[[np.ndarray], float | int]
AggregatesDict = Dict[str, AggregateFunc]

def _safe_std(arr: np.ndarray) -> float:
    n = arr.size
    if n <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))

def _safe_cv(arr: np.ndarray) -> float:
    mean = float(np.mean(arr))
    if mean == 0.0:
        return float('nan')
    return _safe_std(arr) / mean

DEFAULT_AGGREGATES: AggregatesDict = {
    "count": lambda arr: int(arr.size),
    "sum":   lambda arr: float(np.sum(arr)),
    "mean":  lambda arr: float(np.mean(arr)),
    "median": lambda arr: float(np.median(arr)),
    "min":   lambda arr: float(np.min(arr)),
    "max":   lambda arr: float(np.max(arr)),
    "std":   _safe_std,
    "cv":    _safe_cv,
}

# ------------------------------
# Stats builder
# ------------------------------

def _make_stats_row(
    roi_name: str,
    aggregates: AggregatesDict,
    values: np.ndarray | None,
    bounds_error: bool
) -> dict:
    """
    Compute aggregate statistics for an ROI and return as a dictionary.

    High-level: Given a flattened array of pixel values for a single ROI,
    computes user-specified aggregates (e.g., sum, mean, std, cv). 
    If no values are provided (or ROI is out-of-bounds), returns NaN for all aggregates.

    Parameters
    ----------
    roi_name : str
        Name of the ROI.
    aggregates : Dict[str, Callable[[np.ndarray], float | int]]
        Dictionary mapping aggregate names to functions that operate on 1D arrays.
    values : np.ndarray or None
        Flattened array of pixel values from the ROI (may be None if ROI is empty).
    bounds_error : bool
        True if the ROI extended beyond the image bounds (z or xy).

    Returns
    -------
    stats_row : dict
        Dictionary containing the ROI name, bounds_error flag, and computed aggregates.
    """
    row = {"roi": roi_name, "bounds_error": bool(bounds_error)}
    if values is None or values.size == 0:
        for agg_name in aggregates:
            row[agg_name] = float('nan')
        return row

    vals = values.ravel()
    for agg_name, func in aggregates.items():
        row[agg_name] = func(vals)
    return row

# ------------------------------
# Core engine (strict bounds)
# ------------------------------

def getRoiIntensity(
    image: np.ndarray,
    rois: Dict[str, Tuple[int, Polygon]],
    zPlusMinus: int,
    aggregates: AggregatesDict,
) -> list[dict]:
    """
    Compute aggregate statistics for a set of ROIs in a 3D image.

    For each ROI, checks if it is fully within image bounds in x, y, and z.
    If valid, computes specified aggregates (sum, mean, std, etc.) across z slices.
    If ROI is out-of-bounds, returns aggregates as NaN and flags bounds_error=True.

    Parameters
    ----------
    image : np.ndarray
        3D numpy array of shape (z, y, x).
    rois : Dict[str, Tuple[int, Polygon]]
        Dictionary mapping ROI names to a tuple (z_center, polygon).
    zPlusMinus : int
        Number of slices above/below the center slice to include.
    aggregates : AggregatesDict
        Dictionary of aggregate functions to apply to ROI pixel values.

    Returns
    -------
    results : list[dict]
        List of dictionaries, one per ROI, containing 'roi', 'bounds_error', and aggregate values.
    """

    # if image is 2d, add a singleton dimension
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    # assert image.ndim == 3, "image must be 3D (z, y, x)"

    depth, height, width = image.shape

    results: list[dict] = []

    for roi_name, (z_center, geom) in rois.items():
        # Check Z bounds
        z_start = z_center - zPlusMinus
        z_end   = z_center + zPlusMinus + 1
        if z_start < 0 or z_end > depth:
            results.append(_make_stats_row(roi_name, aggregates, None, bounds_error=True))
            continue

        # Check XY bounds
        minx, miny, maxx, maxy = geom.bounds
        if minx < 0 or miny < 0 or maxx > width or maxy > height:
            results.append(_make_stats_row(roi_name, aggregates, None, bounds_error=True))
            continue

        # Convert polygon to integer pixel indices
        x_min = int(np.floor(minx))
        y_min = int(np.floor(miny))
        x_max = int(np.ceil(maxx))
        y_max = int(np.ceil(maxy))

        geom_local = shp_translate(geom, xoff=-x_min, yoff=-y_min)

        mask = rasterize(
            [(geom_local, 1)],
            out_shape=(y_max - y_min, x_max - x_min),
            transform=Affine.identity(),
            fill=0,
            all_touched=True,
            dtype="uint8",
        ).astype(bool)

        sub_arr = image[z_start:z_end, y_min:y_max, x_min:x_max]
        values = sub_arr[:, mask]

        results.append(_make_stats_row(roi_name, aggregates, values, bounds_error=False))

    return results

# ------------------------------
# Multi-channel wrapper
# ------------------------------

def getMultiChannelRoiIntensity(
    channel_images: Dict[str, np.ndarray],  # e.g., {'ch1': arr1, 'ch2': arr2}
    rois: Dict[str, Tuple[int, Polygon]],
    zPlusMinus: int,
    aggregates: AggregatesDict,
) -> pd.DataFrame:
    df_list = []

    for ch_name, img in channel_images.items():
        results = getRoiIntensity(img, rois, zPlusMinus, aggregates)
        df_ch = pd.DataFrame(results)
        rename_dict = {agg: f"{agg}_{ch_name}" for agg in aggregates.keys()}
        df_ch = df_ch.rename(columns=rename_dict)
        df_list.append(df_ch)

    df_merged = df_list[0][['roi', 'bounds_error']].copy()
    for df_ch in df_list:
        for col in df_ch.columns:
            if col not in ['roi', 'bounds_error']:
                df_merged[col] = df_ch[col]

    return df_merged

# ------------------------------
# Visualization
# ------------------------------

def plotRoi(image2d: np.ndarray, rois: Dict[str, Tuple[int, Polygon]], title: str = "ROIs"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image2d, cmap="gray", origin="upper")
    for name, (_, poly) in rois.items():
        if poly.is_empty:
            continue
        x, y = poly.exterior.xy
        ax.plot(x, y, linewidth=2, label=name)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()

# ------------------------------
# Test harness for multi-channel
# ------------------------------

def test_multi_channel_roi():
    # Make two synthetic channels
    nSlices = 10
    imageHeight = 300
    imageWidth = 250
    img_ch1 = _makeCheckerboard(imageShape=(nSlices, imageHeight, imageWidth), squareSize=50, intensity=1)
    img_ch2 = 1 - img_ch1  # inverted channel
    # img_ch3 = 1 - img_ch1  # inverted channel

    channels = {
        'ch1': img_ch1,
        'ch2': img_ch2,
        # 'ch3': img_ch3,
    }

    z_center = img_ch1.shape[0] // 2

    rois: Dict[str, Tuple[int, Polygon]] = {
        "spineHead": (z_center, Polygon([(50, 50), (150, 50), (150, 150), (50, 150)])),
        "spineTail": (z_center, Polygon([(120, 120), (220, 120), (220, 220), (120, 220)])),
        "spine":     (z_center, Polygon([(40, 40), (230, 40), (230, 230), (40, 230)])),
        "spinebackkground": (z_center, Polygon([(-10, -10), (20, -10), (20, 20), (-10, 20)])),
    }

    zPlusMinus = 2
    df_multi = getMultiChannelRoiIntensity(
        channel_images=channels,
        rois=rois,
        zPlusMinus=zPlusMinus,
        aggregates=DEFAULT_AGGREGATES,
    )

    from pprint import pprint
    print("\nMulti-channel ROI intensity results:")
    pprint(df_multi)

    plotRoi(img_ch1[z_center], rois, title=f"ROIs on channel ch1, z={z_center}")


if __name__ == "__main__":
    test_multi_channel_roi()
