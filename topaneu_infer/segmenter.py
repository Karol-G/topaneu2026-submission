"""Binary aneurysm segmentation: preprocessing, sliding-window inference, fold ensembling, export.

The inference procedure follows nnU-Net v2 (Isensee et al., Nature Methods 2021; Apache-2.0), from
which the sliding-window, Gaussian weighting and export logic are taken. Preprocessing per case:

    read -> transpose -> crop to nonzero -> resample to target spacing -> z-score

Note the order: intensities are resampled first and z-scored afterwards, with mean and standard
deviation computed over the cropped, resampled volume. That is how the models were trained.

A model folder holds `model.json` (geometry, architecture, resampling settings) and one
`fold_<k>.safetensors` per ensemble member.
"""
import json
import itertools
from functools import lru_cache
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, gaussian_filter
from safetensors.torch import load_file

from .networks import build_network
from .resampling import compute_new_shape, resample_to_shape

DEFAULT_NUM_THREADS = 8


# ------------------------------------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------------------------------------

def read_image(path) -> Tuple[np.ndarray, sitk.Image, Tuple[float, float, float]]:
    """(1, z, y, x) float32 array, the SimpleITK image (for its geometry), spacing in (z, y, x)."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim != 3:
        raise RuntimeError(f"expected a 3D image, got {arr.ndim}D: {path}")
    sp = img.GetSpacing()
    spacing_zyx = tuple(abs(float(s)) for s in (sp[2], sp[1], sp[0]))
    data = np.vstack([arr[None]], dtype=np.float32, casting='unsafe')
    return data, img, spacing_zyx


# ------------------------------------------------------------------------------------------------
# cropping
# ------------------------------------------------------------------------------------------------

def _bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
    """Half-open [start, stop) bounding box of a 3D mask; the full extent if the mask is empty."""
    out = []
    for ax in range(3):
        other = tuple(a for a in range(3) if a != ax)
        hit = np.flatnonzero(np.any(mask, axis=other))
        out.append([int(hit[0]), int(hit[-1]) + 1] if hit.size else [0, mask.shape[ax]])
    return out


def crop_to_nonzero(data: np.ndarray):
    nonzero = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero |= data[c] != 0
    nonzero = binary_fill_holes(nonzero)
    bbox = _bbox_from_mask(nonzero)
    sl = tuple(slice(*b) for b in bbox)
    return data[(slice(None),) + sl], bbox


# ------------------------------------------------------------------------------------------------
# normalisation
# ------------------------------------------------------------------------------------------------

def zscore_(data: np.ndarray) -> np.ndarray:
    """Per-channel z-score over the whole (cropped, resampled) volume, in place."""
    for c in range(data.shape[0]):
        values = data[c].ravel()
        mean = float(np.mean(values))
        std = float(np.std(values))
        image = data[c].astype(np.float32, copy=False)
        image -= mean
        image /= max(std, 1e-8)
        data[c] = image
    return data


# ------------------------------------------------------------------------------------------------
# sliding window
# ------------------------------------------------------------------------------------------------

@lru_cache(maxsize=2)
def compute_gaussian(tile_size: Tuple[int, ...], sigma_scale: float = 1. / 8, value_scaling_factor: float = 10,
                     dtype=torch.float16, device=torch.device('cpu')) -> torch.Tensor:
    tmp = np.zeros(tile_size)
    tmp[tuple(i // 2 for i in tile_size)] = 1
    g = gaussian_filter(tmp, [i * sigma_scale for i in tile_size], 0, mode='constant', cval=0)
    g /= (np.max(g) / value_scaling_factor)
    g = torch.from_numpy(g).to(device=device, dtype=dtype)
    zero = g == 0
    g[zero] = torch.min(g[~zero])  # never 0, or the normalisation divides by it
    return g


def compute_steps(image_size: Sequence[int], tile_size: Sequence[int], step: float) -> List[List[int]]:
    target = [i * step for i in tile_size]
    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target, tile_size)]
    steps = []
    for d in range(len(tile_size)):
        max_step = image_size[d] - tile_size[d]
        actual = max_step / (num_steps[d] - 1) if num_steps[d] > 1 else 99999999999
        steps.append([int(np.round(actual * i)) for i in range(num_steps[d])])
    return steps


def _pad_to_patch(x: torch.Tensor, patch_size: Sequence[int]):
    """Zero-pad the spatial axes up to the patch size (content centred); returns (padded, slicer)."""
    old = np.array(x.shape)
    new = list(x.shape[:x.ndim - len(patch_size)]) + list(patch_size)
    new = np.array([max(n, o) for n, o in zip(new, old)])
    diff = new - old
    below, above = diff // 2, diff // 2 + diff % 2
    if np.any(diff):
        pad = [int(i) for pair in zip(below[::-1], above[::-1]) for i in pair]
        x = F.pad(x, pad, 'constant', value=0)
    slicer = tuple(slice(int(b), int(s) - int(a)) for b, a, s in zip(below, above, x.shape))
    return x, slicer


# ------------------------------------------------------------------------------------------------
# the model
# ------------------------------------------------------------------------------------------------

class Segmenter:
    """Fold ensemble of one binary segmentation model."""

    def __init__(self, model_dir, folds: Optional[Sequence[int]] = None, device: Optional[torch.device] = None,
                 tile_step_size: float = 0.5, mirror_axes: Optional[Sequence[int]] = None):
        model_dir = Path(model_dir)
        self.cfg = json.loads((model_dir / "model.json").read_text())
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.tile_step_size = tile_step_size
        self.mirror_axes = tuple(mirror_axes) if mirror_axes else None
        self.patch_size = tuple(self.cfg["patch_size"])
        self.num_classes = self.cfg["num_classes"]

        folds = list(folds) if folds is not None else list(self.cfg["folds"])
        tied = self.cfg.get("tied", {})   # keys that alias another tensor (stored once on disk)
        self.weights = []
        for f in folds:
            w = load_file(str(model_dir / f"fold_{f}.safetensors"))
            w.update({k: w[src] for k, src in tied.items()})
            self.weights.append(w)
        self.network = build_network(self.cfg["architecture"], self.cfg["num_input_channels"],
                                     self.num_classes, self.patch_size)
        self.network.load_state_dict(self.weights[0], strict=True)
        # moved once, outside inference mode: parameters created under inference mode cannot be
        # overwritten by the next fold's load_state_dict
        self.network = self.network.to(self.device)
        self.network.eval()

    # ---- preprocessing --------------------------------------------------------------------------

    def preprocess(self, data: np.ndarray, spacing_zyx: Sequence[float]):
        tf = self.cfg["transpose_forward"]
        data = data.astype(np.float32)
        data = data.transpose([0, *[i + 1 for i in tf]])
        spacing = [spacing_zyx[i] for i in tf]
        props = {"spacing_transposed": spacing, "shape_before_cropping": data.shape[1:]}

        data, bbox = crop_to_nonzero(data)
        props["bbox"] = bbox
        props["shape_after_cropping"] = data.shape[1:]

        target = list(self.cfg["spacing"])
        new_shape = compute_new_shape(data.shape[1:], spacing, target)
        data = resample_to_shape(data, new_shape, spacing, target, **self.cfg["resampling"]["image"])
        data = zscore_(data)
        return torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format), props

    # ---- network --------------------------------------------------------------------------------

    @torch.inference_mode()
    def _predict_tile(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.network(x)
        if self.mirror_axes:
            axes = [m + 2 for m in self.mirror_axes]
            combos = [c for i in range(len(axes)) for c in itertools.combinations(axes, i + 1)]
            for ax in combos:
                pred += torch.flip(self.network(torch.flip(x, ax)), ax)
            pred /= (len(combos) + 1)
        return pred

    @torch.inference_mode()
    def _sliding_window(self, data: torch.Tensor) -> torch.Tensor:
        """Gaussian-weighted sliding window; accumulates on the CPU in half precision."""
        autocast = torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" \
            else _NullContext()
        with autocast:
            data, unpad = _pad_to_patch(data, self.patch_size)
            steps = compute_steps(data.shape[1:], self.patch_size, self.tile_step_size)
            slicers = [(slice(None), *[slice(s, s + t) for s, t in zip(pos, self.patch_size)])
                       for pos in itertools.product(*steps)]

            results_device = torch.device("cpu")
            data = data.to(results_device)
            q = Queue(maxsize=2)

            def producer():
                for s in slicers:
                    q.put((torch.clone(data[s][None], memory_format=torch.contiguous_format).to(self.device), s))
                q.put("end")

            Thread(target=producer, daemon=True).start()
            logits = torch.zeros((self.num_classes, *data.shape[1:]), dtype=torch.half, device=results_device)
            n_pred = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
            gaussian = compute_gaussian(self.patch_size, device=results_device)
            while True:
                item = q.get()
                if item == "end":
                    break
                tile, sl = item
                pred = self._predict_tile(tile)[0].to(results_device)
                pred *= gaussian
                logits[sl] += pred
                n_pred[sl[1:]] += gaussian
            torch.div(logits, n_pred, out=logits)
            if torch.any(torch.isinf(logits)):
                raise RuntimeError("inf in the accumulated logits")
            logits = logits[(slice(None), *unpad[1:])]
        return logits

    def predict_logits(self, data: torch.Tensor) -> torch.Tensor:
        n_threads = torch.get_num_threads()
        torch.set_num_threads(min(DEFAULT_NUM_THREADS, n_threads))
        out = None
        for w in self.weights:
            self.network.load_state_dict(w, strict=True)
            p = self._sliding_window(data).to("cpu")
            out = p if out is None else out + p
        if len(self.weights) > 1:
            out /= len(self.weights)
        torch.set_num_threads(n_threads)
        return out

    # ---- export ---------------------------------------------------------------------------------

    def logits_to_mask(self, logits: np.ndarray, props: dict) -> np.ndarray:
        """Resample logits to the cropped input grid, argmax, uncrop, transpose back. (z, y, x) uint8."""
        old_threads = torch.get_num_threads()
        torch.set_num_threads(DEFAULT_NUM_THREADS)
        logits = resample_to_shape(logits, props["shape_after_cropping"], list(self.cfg["spacing"]),
                                   props["spacing_transposed"], **self.cfg["resampling"]["logits"])
        seg = logits.argmax(0)
        full = np.zeros(props["shape_before_cropping"], dtype=np.uint8)
        full[tuple(slice(*b) for b in props["bbox"])] = seg
        torch.set_num_threads(old_threads)
        return full.transpose(self.cfg["transpose_backward"])

    def predict_file(self, image_path, output_path=None) -> sitk.Image:
        """Segment one image file; returns (and optionally writes) a uint8 mask on the input grid."""
        data, img, spacing = read_image(image_path)
        x, props = self.preprocess(data, spacing)
        logits = self.predict_logits(x).cpu().detach().numpy()
        mask = self.logits_to_mask(logits, props)
        out = sitk.GetImageFromArray(mask.astype(np.uint8 if np.max(mask) < 255 else np.uint16, copy=False))
        out.SetSpacing(img.GetSpacing())
        out.SetOrigin(img.GetOrigin())
        out.SetDirection(img.GetDirection())
        if output_path is not None:
            sitk.WriteImage(out, str(output_path), True)
        return out


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
