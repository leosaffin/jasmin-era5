import pathlib

import iris
from iris.cube import CubeList
from iris.util import squeeze
import numpy as np
import pandas as pd


# ERA5 model-level data on jasmin
_here = pathlib.Path(__file__)
era5_path = (
    "/badc/ecmwf-era5/data/oper/an_ml/{year:04d}/{month:02d}/{day:02d}/"
    "ecmwf-era5_oper_an_ml_{year:04d}{month:02d}{day:02d}{hour:02d}00.*.nc"
)


# Downloaded from https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
era5_model_level_info = pd.read_csv(_here.parent / "era5_model_levels.csv", header=0)


def load(time):
    """

    Args:
        time (datetime.datetime):

    Returns:
        iris.cube.CubeList:
    """
    cubes = iris.load(era5_path.format(
        year=time.year,
        month=time.month,
        day=time.day,
        hour=time.hour,
    ))

    # Add bounds to coordinates
    for cube in cubes:
        for axis in ["x", "y"]:
            coord = cube.coord(axis=axis, dim_coords=True)
            if not coord.has_bounds():
                coord.guess_bounds()

    # Squeeze single time dimension from cubes
    cubes = CubeList([squeeze(cube) for cube in cubes])

    return cubes


def p_on_model_levels(lnsp, target_shape):
    """
    Following
    https://confluence.ecmwf.int/display/CKB/
    ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height

    Args:
        lnsp (iris.cube.Cube):
        target_shape (tuple):

    Returns:
        tuple:
            Pressure and pressure thickness on model levels. Both as np.ndarray with
            shape matching target_shape

    """
    # Model levels 1-137 and half levels 0-137
    # Find the dimension for model levels to broadcast to and increase size by one
    target_shape_half_levels = [x for x in target_shape]
    dims = list(range(len(target_shape)))

    unmatched_dimension = [x for x in target_shape if x not in lnsp.shape]
    if len(unmatched_dimension) == 1:
        unmatched_dimension = unmatched_dimension[0]
    else:
        raise ValueError("Target shape must have exactly one extra dimension than lnsp")

    new_dim = dims.pop(target_shape.index(unmatched_dimension))
    target_shape_half_levels[new_dim] += 1

    # Calculate half-level pressure at all grid points
    lnsp = iris.util.broadcast_to_shape(lnsp.data, target_shape_half_levels, dims)
    a = iris.util.broadcast_to_shape(
        np.array(era5_model_level_info["a [Pa]"]), target_shape_half_levels, [new_dim]
    )
    b = iris.util.broadcast_to_shape(
        np.array(era5_model_level_info["b"]), target_shape_half_levels, [new_dim]
    )
    p_half = a + b * np.exp(lnsp)

    # Average half levels to get midpoints (full levels)
    # Create a tuple of slices for indexing along the model_level_number dimension
    sl_n = [slice(None)] * len(target_shape)
    sl_np1 = [slice(None)] * len(target_shape)

    sl_n[new_dim] = slice(None, -1, None)
    sl_np1[new_dim] = slice(1, None, None)

    sl_n = tuple(sl_n)
    sl_np1 = tuple(sl_np1)

    p_ml = 0.5 * (p_half[sl_np1] + p_half[sl_n])
    dp = p_half[sl_np1] - p_half[sl_n]

    return p_ml, dp
