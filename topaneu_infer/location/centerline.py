#!/usr/bin/env python3
"""
Vessel centerline geometry — the shared engine for location-class reconstruction.

Purely geometric: knows nothing about TopAneu label taxonomies. Given a binary tubular mask it
produces a single centerline path, parametrised by *physical* arc length, plus the machinery to
place other structures (branch vessels, aneurysms) onto that parametrisation.

Why a centerline at all: the discriminating question for most location classes is not "does the
lesion touch vessel X" but "where along vessel X does the lesion sit, relative to branch Y".
A binary touch-test cannot express that; an arc-length coordinate can.

Centerline construction:
  1. radius field = EDT inside the mask (spacing-correct), so "central" is measurable
  2. endpoints = the geodesically farthest pair of voxels inside the mask (two-pass farthest-point,
     the standard graph-diameter trick), which for a tube are its two ends
  3. path = minimum-cost route between them with cost 1/radius, which hugs the lumen centre
     instead of cutting corners along the wall
  4. arc length accumulated from actual physical step distances (NOT the routing cost)

Run `python -m topaneu_infer.location.centerline` to execute the synthetic phantom self-tests.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

MIN_VOXELS = 12  # below this a "vessel" is too small for a meaningful centerline


@dataclass
class Centerline:
    """A single vessel centerline, parametrised by physical arc length.

    voxels : (N, 3) int  — voxel indices along the path, ordered
    phys   : (N, 3) float — the same points in physical (mm) coordinates
    arc    : (N,) float  — cumulative arc length in mm; arc[0] == 0
    radius : (N,) float  — local vessel radius (mm) from the EDT
    """
    voxels: np.ndarray
    phys: np.ndarray
    arc: np.ndarray
    radius: np.ndarray

    @property
    def length(self) -> float:
        return float(self.arc[-1])

    def __len__(self) -> int:
        return len(self.arc)

    def flip(self) -> "Centerline":
        """Reverse the parametrisation so arc 0 sits at the other end."""
        return Centerline(
            voxels=self.voxels[::-1].copy(),
            phys=self.phys[::-1].copy(),
            arc=(self.arc[-1] - self.arc[::-1]).copy(),
            radius=self.radius[::-1].copy(),
        )

    def at(self, arc_value: float) -> np.ndarray:
        """Physical point at a given arc length (nearest sample)."""
        return self.phys[int(np.argmin(np.abs(self.arc - arc_value)))]

    def shift(self, offset, spacing) -> "Centerline":
        """Translate into another voxel frame by adding a voxel offset.

        Lets the centerline be computed on a tight crop of the vessel (cheap) and then used in a
        wider window that also contains the neighbouring branch labels. Arc length and radius are
        translation-invariant, so only the coordinates change.
        """
        off = np.asarray(offset, dtype=np.int32)
        vox = self.voxels + off
        return Centerline(voxels=vox,
                          phys=vox * np.asarray(spacing, dtype=float),
                          arc=self.arc.copy(), radius=self.radius.copy())


def _geodesic_farthest(costs: np.ndarray, spacing, seed: Tuple[int, ...]):
    """Farthest reachable voxel from seed under geodesic distance, and the cost map."""
    mcp = MCP_Geometric(costs, sampling=spacing)
    cum, _ = mcp.find_costs([seed])
    finite = np.where(np.isfinite(cum), cum, -1.0)
    idx = np.unravel_index(int(np.argmax(finite)), finite.shape)
    return idx, cum


def extract_centerline(mask: np.ndarray, spacing, min_voxels: int = MIN_VOXELS
                       ) -> Optional[Centerline]:
    """Centerline of a single connected tubular structure. Returns None if unusable.

    `mask` should already be ONE connected component; pass the whole label and it will still work
    but the path will only cover the component containing the geodesic diameter.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < min_voxels:
        return None

    spacing = tuple(float(s) for s in spacing)
    radius = ndimage.distance_transform_edt(mask, sampling=spacing)

    # --- endpoints: two-pass geodesic farthest point (graph-diameter trick) -------------------
    uniform = np.where(mask, 1.0, np.inf)
    seed = tuple(int(c[0]) for c in np.nonzero(mask))
    end_a, _ = _geodesic_farthest(uniform, spacing, seed)
    if not np.isfinite(radius[end_a]) or radius[end_a] <= 0:
        return None
    end_b, _ = _geodesic_farthest(uniform, spacing, end_a)
    if end_a == end_b:
        return None

    # --- route between them, hugging the lumen centre ----------------------------------------
    eps = 1e-6
    centered = np.where(mask, 1.0 / (radius + eps), np.inf)
    mcp = MCP_Geometric(centered, sampling=spacing)
    mcp.find_costs([end_a])
    try:
        path = mcp.traceback(end_b)
    except ValueError:
        return None
    if len(path) < 2:
        return None

    voxels = np.asarray(path, dtype=np.int32)
    phys = voxels * np.asarray(spacing, dtype=float)
    steps = np.linalg.norm(np.diff(phys, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    rad = radius[tuple(voxels.T)]
    return Centerline(voxels=voxels, phys=phys, arc=arc, radius=rad)


def component_nearest(mask: np.ndarray, target: np.ndarray, spacing,
                      min_voxels: int = MIN_VOXELS) -> Optional[np.ndarray]:
    """Of `mask`'s connected components, return the one nearest to `target`.

    Predicted vessel labels are often fragmented; picking the fragment nearest the lesion avoids
    building a centerline on an unrelated piece of the same anatomical label.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    lab, n = ndimage.label(mask)
    if n == 1:
        return mask if mask.sum() >= min_voxels else None

    dist = ndimage.distance_transform_edt(~np.asarray(target, dtype=bool), sampling=spacing)
    best, best_d = None, np.inf
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < min_voxels:
            continue
        d = float(dist[comp].min())
        if d < best_d:
            best, best_d = comp, d
    return best


def project(centerline: Centerline, points_vox: np.ndarray, spacing
            ) -> Tuple[float, float, int]:
    """Place a point set onto the centerline.

    Returns (arc, distance, index): the arc length of the closest centerline sample to the point
    set, the physical distance at that closest approach, and the sample index.

    Used both for aneurysms (-> where the lesion sits) and for branch vessels (-> where the branch
    attaches). In both cases the closest-approach point is the meaningful one.
    """
    pts = np.asarray(points_vox, dtype=float)
    if pts.size == 0:
        return float("nan"), float("inf"), -1
    pts_phys = pts * np.asarray(spacing, dtype=float)
    tree = cKDTree(centerline.phys)
    d, i = tree.query(pts_phys, k=1)
    j = int(np.argmin(d))
    idx = int(i[j])
    return float(centerline.arc[idx]), float(d[j]), idx


def orient_by_reference(centerline: Centerline, reference_vox: np.ndarray, spacing
                        ) -> Tuple[Centerline, bool]:
    """Orient so arc 0 is the end NEAREST the reference structure.

    This is how proximal/distal is established without an atlas: an adjacent labelled vessel of
    known anatomical relationship (e.g. ICA-C1-C5 is proximal to ICA-C6-C7) defines the direction.
    Returns (centerline, flipped).
    """
    pts = np.asarray(reference_vox, dtype=float)
    if pts.size == 0:
        return centerline, False
    pts_phys = pts * np.asarray(spacing, dtype=float)
    tree = cKDTree(pts_phys)
    d_start, _ = tree.query(centerline.phys[0][None, :], k=1)
    d_end, _ = tree.query(centerline.phys[-1][None, :], k=1)
    if float(d_end[0]) < float(d_start[0]):
        return centerline.flip(), True
    return centerline, False


# =================================================================================================
# synthetic phantom self-tests -- analytically known answers
# =================================================================================================

def _tube(shape, spacing, axis_points, radius_mm):
    """Rasterise a tube of given radius around a polyline of voxel coordinates."""
    grid = np.zeros(shape, dtype=bool)
    idx = np.stack(np.indices(shape), axis=-1).astype(float) * np.asarray(spacing, float)
    for p in axis_points:
        c = np.asarray(p, float) * np.asarray(spacing, float)
        grid |= np.linalg.norm(idx - c, axis=-1) <= radius_mm
    return grid


def _check(name, got, want, tol_frac=0.05, tol_abs=0.0):
    tol = max(abs(want) * tol_frac, tol_abs)
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got:.3f}, expected {want:.3f} "
          f"(tol {tol:.3f})")
    return ok


def _z_extent_mm(mask, spacing):
    """Physical span of the mask along z -- the length a correct centerline must cover.

    Note this includes the hemispherical end caps of a rasterised tube: a tube drawn around an
    axis z=a..b with radius r actually occupies z=a-r..b+r, and a centerline that spans the label
    (which is what we want) is correspondingly longer than the axis polyline.
    """
    zs = np.argwhere(mask)[:, 2]
    return float((zs.max() - zs.min()) * spacing[2]), int(zs.min()), int(zs.max())


def _invariants(name, cl: Centerline, mask, spacing):
    """Contract checks that must hold for any centerline, on any input."""
    inside = bool(mask[tuple(cl.voxels.T)].all())
    monotonic = bool(np.all(np.diff(cl.arc) > 0))
    max_step = float(np.max(np.diff(cl.arc)))
    step_limit = 1.75 * float(np.linalg.norm(spacing))  # one 26-connected step, with slack
    contiguous = max_step <= step_limit
    for label, cond in [("all centerline voxels inside mask", inside),
                        ("arc strictly increasing", monotonic),
                        (f"no jumps (max step {max_step:.2f} <= {step_limit:.2f})", contiguous)]:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {label}")
    return inside and monotonic and contiguous


def selftest() -> bool:
    print("centerline synthetic phantom self-tests")
    ok = True

    # ---- 1. straight isotropic tube: centerline must span the label's full extent ------------
    print("\n1. straight tube, isotropic spacing (1,1,1), axis z=5..54, radius 3")
    sp = (1.0, 1.0, 1.0)
    m = _tube((24, 24, 60), sp, [(12, 12, z) for z in range(5, 55)], 3.0)
    span, zmin, zmax = _z_extent_mm(m, sp)
    print(f"     mask occupies z={zmin}..{zmax} -> a spanning centerline measures {span:.1f}mm")
    cl = extract_centerline(m, sp)
    ok &= cl is not None
    ok &= _check("length == label extent", cl.length, span, tol_frac=0.03)
    ok &= _check("mean radius", float(np.mean(cl.radius)), 3.0, tol_frac=0.30)
    ok &= _invariants("straight tube", cl, m, sp)

    # ---- 2. anisotropic spacing: length must follow physical, not voxel, distance ------------
    print("\n2. same tube, anisotropic spacing (1,1,3)")
    sp2 = (1.0, 1.0, 3.0)
    m2 = _tube((24, 24, 60), sp2, [(12, 12, z) for z in range(5, 55)], 3.0)
    span2, _, _ = _z_extent_mm(m2, sp2)
    cl2 = extract_centerline(m2, sp2)
    ok &= cl2 is not None
    ok &= _check("length == label extent (mm)", cl2.length, span2, tol_frac=0.03)
    ok &= _invariants("anisotropic tube", cl2, m2, sp2)

    # ---- 3. curved tube: quarter circle, arc + the two end caps ------------------------------
    print("\n3. quarter-circle tube, arc radius 20, tube radius 3 -> pi/2*20 + 2*3")
    sp3 = (1.0, 1.0, 1.0)
    ang = np.linspace(0, np.pi / 2, 200)
    pts = [(30 + 20 * np.cos(t), 30 + 20 * np.sin(t), 12) for t in ang]
    m3 = _tube((56, 56, 24), sp3, pts, 3.0)
    cl3 = extract_centerline(m3, sp3)
    ok &= cl3 is not None
    ok &= _check("arc length", cl3.length, np.pi / 2 * 20 + 2 * 3, tol_frac=0.10)
    ok &= _invariants("curved tube", cl3, m3, sp3)

    # ---- 4. projection: exact arc of a structure attached at a known z -----------------------
    print("\n4. projection of a structure attached at a known height")
    cl_o, _ = orient_by_reference(cl, np.argwhere(_tube(m.shape, sp, [(12, 12, 0)], 2.0)), sp)
    z_blob = 39
    blob = np.zeros_like(m)
    blob[12 + 3: 12 + 7, 12, z_blob] = True   # sticks out sideways at that height
    arc_o, dist, _ = project(cl_o, np.argwhere(blob), sp)
    # with arc 0 at the proximal cap (zmin), the expected arc is simply the z offset
    ok &= _check("projected arc", arc_o, (z_blob - zmin) * sp[2], tol_abs=1.5)
    ok &= _check("closest-approach distance", dist, 3.0, tol_abs=1.5)

    # ---- 5. orientation by reference ---------------------------------------------------------
    print("\n5. orientation: reference blob at the z=5 end must become arc 0")
    ref_low = np.argwhere(_tube(m.shape, sp, [(12, 12, 2)], 2.0))
    ref_high = np.argwhere(_tube(m.shape, sp, [(12, 12, 57)], 2.0))
    cl_low, _ = orient_by_reference(cl, ref_low, sp)
    cl_high, _ = orient_by_reference(cl, ref_high, sp)
    z_start_low = cl_low.voxels[0][2]
    z_start_high = cl_high.voxels[0][2]
    print(f"  [{'PASS' if z_start_low < 15 else 'FAIL'}] arc0 near z=5 when ref is low "
          f"(got z={z_start_low})")
    print(f"  [{'PASS' if z_start_high > 44 else 'FAIL'}] arc0 near z=54 when ref is high "
          f"(got z={z_start_high})")
    ok &= bool(z_start_low < 15) and bool(z_start_high > 44)
    ok &= _check("flip preserves length", cl_low.length, cl_high.length, tol_frac=0.001)

    # ---- 6. branch landmark at a known position ---------------------------------------------
    print("\n6. side branch attached at a known height -> exact landmark arc")
    z_branch = 24
    branch = _tube(m.shape, sp, [(12 + d, 12, z_branch) for d in range(3, 10)], 2.0)
    arc_b, dist_b, _ = project(cl_low, np.argwhere(branch), sp)
    ok &= _check("landmark arc", arc_b, (z_branch - zmin) * sp[2], tol_abs=1.5)

    # a landmark's arc must be invariant to which end the parametrisation starts from
    arc_b_flipped, _, _ = project(cl_low.flip(), np.argwhere(branch), sp)
    ok &= _check("landmark arc under flip", cl_low.length - arc_b_flipped, arc_b, tol_abs=0.01)

    # ---- 7. degenerate inputs must be handled, not crash ------------------------------------
    print("\n7. degenerate inputs")
    for name, arr in [("empty", np.zeros((10, 10, 10), bool)),
                      ("single voxel", None),
                      ("tiny blob", None)]:
        if arr is None:
            arr = np.zeros((10, 10, 10), bool)
            if name == "single voxel":
                arr[5, 5, 5] = True
            else:
                arr[4:6, 4:6, 4:6] = True
        r = extract_centerline(arr, sp)
        print(f"  [{'PASS' if r is None else 'FAIL'}] {name} -> None (got {type(r).__name__})")
        ok &= r is None

    # ---- 8. fragmented label: component_nearest must pick the right fragment -----------------
    print("\n8. fragmented label -> component nearest the target is selected")
    frag = np.zeros((24, 24, 60), bool)
    frag |= _tube(frag.shape, sp, [(12, 12, z) for z in range(2, 18)], 3.0)    # near
    frag |= _tube(frag.shape, sp, [(12, 12, z) for z in range(40, 56)], 3.0)   # far
    target = np.zeros_like(frag); target[12, 12, 0] = True
    picked = component_nearest(frag, target, sp)
    zs = np.argwhere(picked)[:, 2]
    ok &= bool(zs.max() < 30)
    print(f"  [{'PASS' if zs.max() < 30 else 'FAIL'}] picked the near fragment "
          f"(z range {zs.min()}..{zs.max()})")

    print(f"\n{'ALL SELF-TESTS PASSED' if ok else 'SOME SELF-TESTS FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
