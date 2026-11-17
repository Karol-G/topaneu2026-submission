"""Per-aneurysm geometry: parent vessel, position along its centerline, neighbouring branches.

For every connected component of the binary aneurysm mask:

  1. parent vessel   = the vessel label closest to the component (within PARENT_SEARCH_MM)
  2. centerline      = of the parent label's fragment nearest the component
  3. orientation     = proximal -> distal, fixed by an adjacent vessel of known relationship
  4. position        = arc length (mm and fraction) of the component along the centerline
  5. landmarks       = arc position of every other vessel label attaching near the centerline

The records feed `rules.decide`, which maps them to one of the 52 location classes.
"""
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage

from .centerline import component_nearest, extract_centerline, orient_by_reference, project

PARENT_SEARCH_MM = 10.0   # how far to look for a parent vessel
ROI_MARGIN_MM = 25.0      # working window around the parent label
LANDMARK_MAX_MM = 15.0    # a branch further than this from the centerline is not a landmark

# How to orient each parent vessel's centerline: which neighbouring label marks the PROXIMAL end
# (role 'proximal') or the DISTAL end (role 'distal', i.e. flip after matching).
# side: 'same' = same laterality as the parent, 'any' = either/unsided.
ORIENTATION = {
    "ICA-C6-C7": ("ICA-C1-C5", "same", "proximal"),
    "ICA-C1-C5": ("ICA-C6-C7", "same", "distal"),
    "A1A2":      ("ICA-C6-C7", "same", "proximal"),
    "A3":        ("A1A2",      "same", "proximal"),
    "M1":        ("ICA-C6-C7", "same", "proximal"),
    "M2":        ("M1",        "same", "proximal"),
    "M3":        ("M2",        "same", "proximal"),
    "BA":        ("VA",        "any",  "proximal"),
    "VA":        ("BA",        "any",  "distal"),
    "P1P2":      ("BA",        "any",  "proximal"),
    "P3P4":      ("P1P2",      "same", "proximal"),
    "SCA":       ("BA",        "any",  "proximal"),
    "AICA":      ("BA",        "any",  "proximal"),
    "PICA":      ("VA",        "same", "proximal"),
    "Pcom":      ("ICA-C6-C7", "same", "proximal"),
    "AChA":      ("ICA-C6-C7", "same", "proximal"),
    "OA":        ("ICA-C6-C7", "same", "proximal"),
}


def vessel_family(name: str):
    if name.startswith("R-"):
        return name[2:], "R"
    if name.startswith("L-"):
        return name[2:], "L"
    return name, None


def component_features(aneurysm: np.ndarray, vessel: np.ndarray, spacing,
                       ves_ids: Dict[str, int]) -> Tuple[List[dict], np.ndarray, int]:
    """Geometry records for every aneurysm component.

    `aneurysm` is a boolean mask, `vessel` the multi-class vessel mask on the same grid, `spacing`
    the voxel size in mm along the array axes, `ves_ids` the vessel name -> label id mapping.
    Returns (records, labeled components, number of components).
    """
    id_to_ves = {v: k for k, v in ves_ids.items()}
    present = {id_to_ves[int(i)] for i in np.unique(vessel) if int(i) != 0}
    labeled, n = ndimage.label(aneurysm)

    out = []
    centerline_cache = {}
    for comp_id in range(1, n + 1):
        rec = {"component": comp_id, "vessels_present": sorted(present)}
        comp = labeled == comp_id
        rec["n_voxels"] = int(comp.sum())

        # ---- ROI around the aneurysm, wide enough to find a parent ---------------------------
        obj = ndimage.find_objects(comp.astype(np.uint8))[0]
        mar = [int(np.ceil(PARENT_SEARCH_MM / s)) + 2 for s in spacing]
        sl = tuple(slice(max(0, o.start - m), min(d, o.stop + m))
                   for o, m, d in zip(obj, mar, comp.shape))
        dist_a = ndimage.distance_transform_edt(~comp[sl], sampling=spacing)
        ves_roi = vessel[sl]

        cands = []
        for vid in (int(i) for i in np.unique(ves_roi) if int(i) != 0):
            d = float(dist_a[ves_roi == vid].min())
            if d <= PARENT_SEARCH_MM:
                cands.append((d, vid))
        if not cands:
            rec["parent"] = None
            rec["note"] = "no vessel within parent search radius"
            out.append(rec)
            continue

        cands.sort()
        rec["parent_dist_mm"] = round(cands[0][0], 3)
        parent_name = id_to_ves[cands[0][1]]
        rec["parent"] = parent_name
        rec["candidates"] = [{"vessel": id_to_ves[v], "dist_mm": round(d, 3)} for d, v in cands]
        p_fam, p_side = vessel_family(parent_name)

        # ---- working window around the PARENT label, for its centerline ----------------------
        pmask_full = vessel == cands[0][1]
        pobj = ndimage.find_objects(pmask_full.astype(np.uint8))[0]
        pmar = [int(np.ceil(ROI_MARGIN_MM / s)) + 2 for s in spacing]
        psl = tuple(slice(max(0, o.start - m), min(d, o.stop + m))
                    for o, m, d in zip(pobj, pmar, pmask_full.shape))

        key = (parent_name, tuple((s.start, s.stop) for s in psl))
        if key not in centerline_cache:
            pm = component_nearest(pmask_full[psl], comp[psl], spacing)
            cl_built = None
            if pm is not None:
                # build the centerline on a tight crop of the vessel component -- the path search
                # is O(window), and the padded landmark window is many times larger
                tobj = ndimage.find_objects(pm.astype(np.uint8))[0]
                tsl = tuple(slice(max(0, o.start - 1), min(d, o.stop + 1))
                            for o, d in zip(tobj, pm.shape))
                cl_tight = extract_centerline(pm[tsl], spacing)
                if cl_tight is not None:
                    cl_built = cl_tight.shift([s.start for s in tsl], spacing)
            centerline_cache[key] = cl_built
        cl = centerline_cache[key]
        if cl is None:
            rec["note"] = "centerline extraction failed"
            out.append(rec)
            continue

        ves_win = vessel[psl]

        # ---- orient the centerline ----------------------------------------------------------
        oriented = False
        if p_fam in ORIENTATION:
            ref_fam, side_mode, role = ORIENTATION[p_fam]
            ref_name = ref_fam if side_mode == "any" or p_side is None else f"{p_side}-{ref_fam}"
            ref_ids = []
            if side_mode == "any":
                ref_ids = [v for k, v in ves_ids.items() if vessel_family(k)[0] == ref_fam]
            elif ref_name in ves_ids:
                ref_ids = [ves_ids[ref_name]]
            ref_mask = np.isin(ves_win, ref_ids) if ref_ids else None
            if ref_mask is not None and ref_mask.any():
                cl, flipped = orient_by_reference(cl, np.argwhere(ref_mask), spacing)
                if role == "distal":
                    cl = cl.flip()
                oriented = True
                rec["orient_ref"] = ref_name
                rec["orient_role"] = role
        rec["oriented"] = oriented
        rec["centerline_len_mm"] = round(cl.length, 3)
        rec["centerline_n"] = len(cl)

        # ---- where the aneurysm sits --------------------------------------------------------
        arc, d_cl, idx = project(cl, np.argwhere(comp[psl]), spacing)
        rec["aneurysm_arc_mm"] = round(arc, 3)
        rec["aneurysm_arc_frac"] = round(arc / cl.length, 4) if cl.length > 0 else None
        rec["aneurysm_dist_to_centerline_mm"] = round(d_cl, 3)
        rec["local_radius_mm"] = round(float(cl.radius[idx]), 3) if idx >= 0 else None

        # ---- where every neighbouring branch attaches ---------------------------------------
        landmarks = {}
        for vid in (int(i) for i in np.unique(ves_win) if int(i) != 0):
            name = id_to_ves[vid]
            if name == parent_name:
                continue
            l_arc, l_dist, _ = project(cl, np.argwhere(ves_win == vid), spacing)
            if l_dist <= LANDMARK_MAX_MM:
                landmarks[name] = {
                    "arc_mm": round(l_arc, 3),
                    "arc_frac": round(l_arc / cl.length, 4) if cl.length > 0 else None,
                    "dist_to_centerline_mm": round(l_dist, 3),
                    "delta_mm": round(arc - l_arc, 3),      # + => aneurysm distal to landmark
                }
        rec["landmarks"] = landmarks
        out.append(rec)

    return out, labeled, n
