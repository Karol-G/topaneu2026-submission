"""Image and logit resampling.

Taken from nnU-Net v2 (`nnunetv2/preprocessing/resampling/default_resampling.py`, Apache-2.0),
reduced to the image path: spline resampling with skimage, and "separate z" handling for
anisotropic volumes (in-plane spline, nearest neighbour along the low-resolution axis).
"""
from copy import deepcopy
from typing import List, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import map_coordinates
from skimage.transform import resize

ANISO_THRESHOLD = 3  # spacing ratio above which a volume counts as anisotropic


def get_do_separate_z(spacing: Sequence[float], anisotropy_threshold=ANISO_THRESHOLD) -> bool:
    spacing = np.asarray(spacing, dtype=float)
    return (np.max(spacing) / np.min(spacing)) > anisotropy_threshold


def get_lowres_axis(spacing: Sequence[float]) -> np.ndarray:
    spacing = np.asarray(spacing, dtype=float)
    return np.where(np.max(spacing) / spacing == 1)[0]


def compute_new_shape(old_shape: Sequence[int], old_spacing: Sequence[float],
                      new_spacing: Sequence[float]) -> np.ndarray:
    assert len(old_spacing) == len(old_shape) == len(new_spacing)
    return np.array([int(round(float(i) / float(j) * k))
                     for i, j, k in zip(old_spacing, new_spacing, old_shape)])


def determine_do_sep_z_and_axis(force_separate_z: Union[bool, None], current_spacing: Sequence[float],
                                new_spacing: Sequence[float],
                                separate_z_anisotropy_threshold: float = ANISO_THRESHOLD
                                ) -> Tuple[bool, Union[int, None]]:
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        axis = get_lowres_axis(current_spacing) if force_separate_z else None
    elif get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
        do_separate_z, axis = True, get_lowres_axis(current_spacing)
    elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
        do_separate_z, axis = True, get_lowres_axis(new_spacing)
    else:
        do_separate_z, axis = False, None

    if axis is not None:
        if len(axis) in (2, 3):
            # e.g. (0.24, 1.25, 1.25): two low-res axes, no single out-of-plane axis
            do_separate_z, axis = False, None
        else:
            axis = axis[0]
    return do_separate_z, axis


def resample_to_shape(data: np.ndarray, new_shape: Sequence[int], current_spacing: Sequence[float],
                      new_spacing: Sequence[float], order: int = 3, order_z: int = 0,
                      force_separate_z: Union[bool, None] = None) -> np.ndarray:
    """Resample a (c, x, y, z) array to `new_shape`. Output keeps the input dtype."""
    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing)
    return _resample(data, new_shape, axis, order, do_separate_z, order_z)


def _resample(data: np.ndarray, new_shape: Union[List[int], np.ndarray], axis: Union[None, int],
              order: int, do_separate_z: bool, order_z: int) -> np.ndarray:
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == data.ndim - 1

    kwargs = {'mode': 'edge', 'anti_aliasing': False}
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if not np.any(shape != new_shape):
        return data

    dtype_out = data.dtype
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
    data = data.astype(float, copy=False)
    if not do_separate_z:
        for c in range(data.shape[0]):
            reshaped_final[c] = resize(data[c], new_shape, order, **kwargs)
        return reshaped_final

    assert axis is not None, 'separate z resampling needs the anisotropic axis'
    if axis == 0:
        new_shape_2d = new_shape[1:]
    elif axis == 1:
        new_shape_2d = new_shape[[0, 2]]
    else:
        new_shape_2d = new_shape[:-1]

    for c in range(data.shape[0]):
        tmp = deepcopy(new_shape)
        tmp[axis] = shape[axis]
        reshaped_here = np.zeros(tmp)
        for slice_id in range(shape[axis]):
            if axis == 0:
                reshaped_here[slice_id] = resize(data[c, slice_id], new_shape_2d, order, **kwargs)
            elif axis == 1:
                reshaped_here[:, slice_id] = resize(data[c, :, slice_id], new_shape_2d, order, **kwargs)
            else:
                reshaped_here[:, :, slice_id] = resize(data[c, :, :, slice_id], new_shape_2d, order, **kwargs)
        if shape[axis] != new_shape[axis]:
            # nearest-neighbour (order_z) along the low-resolution axis, align_corners=False
            rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
            orig_rows, orig_cols, orig_dim = reshaped_here.shape
            row_scale = float(orig_rows) / rows
            col_scale = float(orig_cols) / cols
            dim_scale = float(orig_dim) / dim
            map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
            map_rows = row_scale * (map_rows + 0.5) - 0.5
            map_cols = col_scale * (map_cols + 0.5) - 0.5
            map_dims = dim_scale * (map_dims + 0.5) - 0.5
            coord_map = np.array([map_rows, map_cols, map_dims])
            reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode='nearest')[None]
        else:
            reshaped_final[c] = reshaped_here
    return reshaped_final
