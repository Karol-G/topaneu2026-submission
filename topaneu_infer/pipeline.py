"""The full chain for one case.

    image ──► binary aneurysm model (fold ensemble, per modality) ──► binary mask ─┐
                                                                                  ├─► location
    image ──► TA36 vessel model (organizers' release, subprocess) ──► vessel mask ─┘   classes

Model folder layout (`model_dir`):

    ct/ mr/          model.json + fold_<k>.safetensors     binary aneurysm models
    vessel/          the organizers' TA36 vessel checkpoint
    mappings/        vessel_mapping.json, location_mapping.json
    params.json      location rule thresholds
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import nibabel as nib
import SimpleITK as sitk
import torch

from .location import classify_locations, load_mappings, load_params
from .segmenter import Segmenter

VESSEL_MODEL = "Dataset572_TopAneu_Vessel_36fgCls_wLRSwap"
VESSEL_SUBDIR = "resEncM"   # the single vessel model used (prefix of its folder name)


def _log(msg, t0=None):
    el = f"  [{time.time() - t0:6.1f}s]" if t0 else ""
    print(f"[topaneu]{el} {msg}", flush=True)


def segment_aneurysm(image_file: Path, out_file: Path, model_dir: Path,
                     folds: Optional[Sequence[int]] = None) -> Path:
    seg = Segmenter(model_dir, folds=folds)
    seg.predict_file(image_file, out_file)
    del seg
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_file


def segment_vessels(in_dir: Path, out_dir: Path, model_dir: Path, vessel_code: Path) -> Path:
    """The organizers' TA36 vessel model, run with its own code in a subprocess."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["TOPANEU_MODEL_ROOT"] = str(model_dir / "vessel" / VESSEL_MODEL)
    env["TOPANEU_VESSEL_MODELS"] = VESSEL_SUBDIR
    env["PYTHONPATH"] = str(vessel_code)
    cmd = [sys.executable, str(vessel_code / "run_inference.py"),
           "-i", str(in_dir), "-o", str(out_dir), "--suffix", "_0000.nii.gz", "--fuse_logits"]
    r = subprocess.run(cmd, cwd=str(vessel_code), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"vessel stage exited {r.returncode}")
    produced = sorted(out_dir.glob("*.nii.gz"))
    if not produced:
        raise RuntimeError(f"vessel stage produced nothing in {out_dir}")
    return produced[0]


def run_case(img: sitk.Image, modality: str, model_dir, vessel_code,
             folds: Optional[Sequence[int]] = None, work_dir=None) -> Tuple[sitk.Image, List[int]]:
    """(image, 'ct'|'mr') -> (52-class uint8 mask on the input grid, detected location class ids).

    `work_dir` keeps every intermediate (input, binary mask, vessel mask); by default a temporary
    directory is used and removed.
    """
    model_dir, vessel_code = Path(model_dir), Path(vessel_code)
    t0 = time.time()
    keep = work_dir is not None
    work = Path(work_dir) if keep else Path(tempfile.mkdtemp(prefix="topaneu_"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        in_dir = work / "in"
        in_dir.mkdir(exist_ok=True)
        image_file = in_dir / "case_0000.nii.gz"
        sitk.WriteImage(img, str(image_file), useCompression=True)
        _log(f"input {img.GetSize()} spacing {tuple(round(s, 3) for s in img.GetSpacing())}", t0)

        (work / "aneurysm").mkdir(exist_ok=True)
        bin_file = segment_aneurysm(image_file, work / "aneurysm" / "case.nii.gz",
                                    model_dir / modality, folds)
        _log(f"aneurysm mask done ({modality})", t0)

        ves_file = segment_vessels(in_dir, work / "vessel", model_dir, vessel_code)
        _log("vessel mask done", t0)

        ves_ids, loc_ids = load_mappings(model_dir / "mappings")
        params = load_params(model_dir / "params.json")
        pred_vol, detected, rows, nii = classify_locations(bin_file, ves_file, ves_ids, loc_ids, params)
        _log(f"{len(rows)} component(s) -> classes {detected or 'none'}", t0)

        # back to SimpleITK through NIfTI, so the axis order is handled by the file format
        tmp_out = work / "locations.nii.gz"
        nib.save(nib.Nifti1Image(pred_vol, nii.affine, nii.header), str(tmp_out))
        out = sitk.Cast(sitk.ReadImage(str(tmp_out)), sitk.sitkUInt8)
        out.CopyInformation(img)
        return out, detected
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
