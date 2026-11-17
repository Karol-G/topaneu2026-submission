"""Aneurysm location classification from a binary aneurysm mask and a multi-class vessel mask.

    binary aneurysm mask ─┐
                          ├─► per-component geometry (features.py) ─► decision rules (rules.py)
    36-class vessel mask ─┘                                              │
                                                                         ▼
                                              52-class location mask + list of detected classes
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .features import component_features
from .rules import DEFAULTS, decide


def load_params(path) -> dict:
    """Thresholds from a params file (flat, or nested under "params"). Unknown keys are an error."""
    raw = json.loads(Path(path).read_text())
    p = raw["params"] if isinstance(raw.get("params"), dict) else raw
    unknown = sorted(set(p) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"{path}: unknown threshold(s) {unknown}")
    return p


def load_mappings(mappings_dir):
    """(vessel name -> id, location class name -> id)."""
    mappings_dir = Path(mappings_dir)
    ves = json.loads((mappings_dir / "vessel_mapping.json").read_text())["labels"]
    loc = json.loads((mappings_dir / "location_mapping.json").read_text())["labels"]
    return ves, loc


def classify_locations(aneurysm_path, vessel_path, ves_ids: dict, loc_ids: dict, params: dict):
    """Returns (52-class volume, sorted detected class ids, per-component rows, aneurysm NIfTI).

    The volume is in the aneurysm mask's NIfTI array order; write it with that image's affine.
    """
    a_img = nib.load(str(aneurysm_path))
    aneurysm = np.asarray(a_img.dataobj) > 0
    vessel = np.asarray(nib.load(str(vessel_path)).dataobj)
    if vessel.shape != aneurysm.shape:
        raise RuntimeError(f"shape mismatch: aneurysm {aneurysm.shape} vs vessel {vessel.shape}")
    spacing = tuple(float(x) for x in a_img.header.get_zooms()[:3])

    recs, labeled, _ = component_features(aneurysm, vessel, spacing, ves_ids)

    pred_vol = np.zeros(labeled.shape, dtype=np.uint8)
    rows = []
    for rec in recs:
        cid = rec["component"]
        pred, reason = decide(rec, params, loc_ids)
        if pred:
            pred_vol[labeled == cid] = pred
        rows.append({"component": cid, "pred": int(pred), "parent": rec.get("parent"),
                     "reason": reason})
    detected = sorted({r["pred"] for r in rows if r["pred"]})
    return pred_vol, detected, rows, a_img
