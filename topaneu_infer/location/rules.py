"""Location-class decision rules: one geometry record -> one of the 52 location classes.

Each aneurysm is decided by the family of its parent vessel, using its arc position along that
vessel's centerline (`aneurysm_arc_frac`, 0 = proximal, 1 = distal) and the arc offsets of nearby
branch origins (`landmarks[...]["delta_mm"]`, positive = aneurysm distal to the branch).

DEFAULTS holds every threshold; `params_challenge.json` overrides the ones that were fitted on the
training data. ABSTAIN (0) means no class is assigned and the component is left out of the output.
"""

ABSTAIN = 0


def family(name):
    if name.startswith("R-"):
        return name[2:], "R"
    if name.startswith("L-"):
        return name[2:], "L"
    return name, None


DEFAULTS = {
    "ica_term_mm": 4.5,
    "ica_term_frac": 0.6,
    "ica_c6_split_mm": 6.0,
    "ica_frac_split": 0.35,
    "ica_junction_mm": 2.5,
    "ica_infra_frac": 0.06,
    "ica_c7_default": "nonBranch",
    "ica_c6_default": "nonOA",
    "m1_junction_frac": 0.65,
    "m1_trunk_frac": 0.0,
    "m1_radius_mm": 0.0,
    "m1_radius_min_frac": 0.0,
    "m2_distal_frac": 0.8,
    "m2_default": "junction",
    "ba_prox_frac": 0.35,
    "ba_tip_frac": 0.75,
    "ba_junction_mm": 3.0,
    "ba_default": "trunk",
    "ba_min_len_mm": 0.0,
    "ba_use_type": "off",
    "va_vaba_frac": 0.88,
    "va_pica_frac": 0.76,
    "va_junction_mm": 6.0,
    "va_default": "trunk",
    "va_use_type": "off",
    "a1a2_use_acom": True,
    "a1a2_acom_mm": 3.0,
    "a1a2_a1_frac": 0.2,
    "a1a2_acom_frac": 0.46,
}


def _lm(rec, fam, side=None, max_dist=6.0):
    """Landmark record for a vessel family (optionally side-specific), or None if not usable."""
    lms = rec.get("landmarks") or {}
    key = f"{side}-{fam}" if side else fam
    d = lms.get(key)
    if d is None or d.get("dist_to_centerline_mm", 1e9) > max_dist:
        return None
    return d


def _delta(rec, fam, side=None, max_dist=6.0):
    d = _lm(rec, fam, side, max_dist)
    return None if d is None else d["delta_mm"]


def _nearest_other_family(rec, exclude):
    """The nearest candidate vessel whose family is not `exclude`, as (name, family, side).

    Used only by the fragment guards, where the parent label itself is judged untrustworthy and the
    lesion has to be handed to a neighbouring vessel. Deliberately returns only the *nearest* such
    candidate rather than searching for a preferred family anywhere in the list: `candidates` holds
    everything within the 10 mm parent-search radius, so a search-by-family would happily reach past
    two closer vessels to grab a distant one that happens to suit the rule.
    """
    for c in rec.get("candidates") or []:
        f, s = family(c["vessel"])
        if f != exclude:
            return c["vessel"], f, s
    return None, None, None


def _any_side_delta(rec, fam, max_dist=6.0):
    """Smallest-|delta| landmark of this family on either side, as (delta, side)."""
    lms = rec.get("landmarks") or {}
    best = None
    for key, d in lms.items():
        f, s = family(key)
        if f != fam or d.get("dist_to_centerline_mm", 1e9) > max_dist:
            continue
        if best is None or abs(d["delta_mm"]) < abs(best[0]):
            best = (d["delta_mm"], s)
    return best if best else (None, None)


def decide(rec, params, L):
    """Return (location_class_id, reason). ABSTAIN (0) means 'no confident assignment'.

    `params` overrides DEFAULTS; `L` maps location class names to ids (location_mapping.json).
    """
    P = dict(DEFAULTS)
    if params:
        P.update(params)

    parent = rec.get("parent")
    if not parent:
        return ABSTAIN, "no parent vessel"
    if not rec.get("centerline_len_mm"):
        return ABSTAIN, "no centerline"
    af = rec.get("aneurysm_arc_frac")
    if af is None:
        return ABSTAIN, "no arc position"
    fam, side = family(parent)
    S = side

    # ---- internal carotid, C6-C7: the largest and most heterogeneous family -----------------
    if fam == "ICA-C6-C7":
        d_m1 = _delta(rec, "M1", S)
        d_a1a2 = _delta(rec, "A1A2", S)
        d_oa = _delta(rec, "OA", S)
        d_pcom = _delta(rec, "Pcom", S)
        d_acha = _delta(rec, "AChA", S)

        # 1. carotid terminus -- the cleanest signal in the whole dataset: the lesion sits where
        #    both daughter vessels attach (delta ~0 to M1 AND A1A2)
        near_m1 = d_m1 is not None and abs(d_m1) <= P["ica_term_mm"]
        near_a1a2 = d_a1a2 is not None and abs(d_a1a2) <= P["ica_term_mm"]
        if (near_m1 or near_a1a2) and af >= P["ica_term_frac"]:
            return L[f"{S}-3.7 ICA C7-terminus"], "terminus (at M1/A1A2 attachment)"

        # 2. C6 vs C7 territory. The OA origin IS the C5/C6 boundary anatomically, so its
        #    arc offset is the natural discriminator; arc_frac is the fallback when OA is missing.
        if d_oa is not None:
            in_c6 = d_oa <= P["ica_c6_split_mm"]
            basis = "OA offset"
        else:
            in_c6 = af <= P["ica_frac_split"]
            basis = "arc_frac (OA missing)"

        if in_c6:
            if af <= P["ica_infra_frac"]:
                return L[f"{S}-3.1 ICA infraclinoid C1-C5"], f"very proximal on C6C7 [{basis}]"
            if d_oa is not None and abs(d_oa) <= P["ica_junction_mm"]:
                return L[f"{S}-3.2 ICA C6-OA-junction"], "at OA origin"
            if P["ica_c6_default"] == "OA":
                return L[f"{S}-3.2 ICA C6-OA-junction"], f"C6 default [{basis}]"
            return L[f"{S}-3.3 ICA C6-nonOA"], f"C6 away from OA [{basis}]"  # tier B

        # 3. within C7: pick whichever branch landmark is closest, if any is close enough.
        #    Pcom and AChA origins are only ~2-4mm apart, so nearest-wins is the honest rule.
        cands = []
        if d_pcom is not None:
            cands.append((abs(d_pcom), L[f"{S}-3.4 ICA C7-Pcom-junction"], "at Pcom origin"))
        if d_acha is not None:
            cands.append((abs(d_acha), L[f"{S}-3.5 ICA C7-AChA-junction"], "at AChA origin"))
        cands.sort()
        if cands and cands[0][0] <= P["ica_junction_mm"]:
            return cands[0][1], cands[0][2]
        if P["ica_c7_default"] == "Pcom":
            return L[f"{S}-3.4 ICA C7-Pcom-junction"], f"C7 default=Pcom [{basis}]"
        return L[f"{S}-3.6 ICA C7-nonBranch"], f"C7 away from branches [{basis}]"  # tier B

    # ---- internal carotid, C1-C5 -------------------------------------------------------------
    if fam == "ICA-C1-C5":
        return L[f"{S}-3.1 ICA infraclinoid C1-C5"], "parented on C1-C5"

    # ---- middle cerebral ---------------------------------------------------------------------
    if fam == "M1":
        if af >= P["m1_junction_frac"]:
            return L[f"{S}-5.3 M1-M2 junction"], "distal M1 (main bifurcation)"
        if af <= P["m1_trunk_frac"]:
            return L[f"{S}-5.1 M1 trunk"], "proximal M1"
        # arc-ambiguous band: consult the radius collapse as a second opinion
        r_loc = rec.get("local_radius_mm")
        if (P["m1_radius_mm"] > 0 and r_loc is not None and r_loc <= P["m1_radius_mm"]
                and af >= P["m1_radius_min_frac"]):
            return L[f"{S}-5.3 M1-M2 junction"], "mid M1 but the lumen has tapered (bifurcation)"
        return L[f"{S}-5.2 M1 early bifurcation"], "mid M1 (early bifurcation)"

    if fam in ("M2", "M3"):
        d_m1 = _delta(rec, "M1", S)
        if af >= P["m2_distal_frac"] and d_m1 is None:
            return L[f"{S}-5.3 Distal-M2M3"], "distal M2/M3, no M1 in reach"
        if P["m2_default"] == "distal":
            return L[f"{S}-5.3 Distal-M2M3"], "M2 default=distal"
        return L[f"{S}-5.3 M1-M2 junction"], "proximal M2 (at the bifurcation)"

    # ---- vertebrobasilar ---------------------------------------------------------------------
    if fam == "BA":
        d_p1p2, _ = _any_side_delta(rec, "P1P2")
        d_sca, sca_side = _any_side_delta(rec, "SCA")
        d_aica, aica_side = _any_side_delta(rec, "AICA")
        d_va, _ = _any_side_delta(rec, "VA")

        # fragment guard: a BA label a few mm long is not the basilar artery
        if 0 < P["ba_min_len_mm"] and rec["centerline_len_mm"] < P["ba_min_len_mm"]:
            _, alt_fam, alt_side = _nearest_other_family(rec, "BA")
            if alt_fam == "VA":
                return L[f"{alt_side}-1.1 VA trunk"], "BA label is a fragment -> nearest VA"

        if af <= P["ba_prox_frac"]:
            return L["1.5 VA-BA junction"], "proximal BA"
        if P["ba_use_type"] != "off" and rec.get("aneurysm_type") in ("dissecting", "fusiform"):
            # segmental wall disease: not a tip or branch-point lesion
            return L["1.4 BA trunk"], "BA, non-saccular: segmental wall disease => trunk"
        if af >= P["ba_tip_frac"]:
            # tip vs SCA origin: they are anatomically adjacent, so require the SCA to be the
            # clearly closer landmark before preferring it over the tip
            if (d_sca is not None and abs(d_sca) <= P["ba_junction_mm"]
                    and (d_p1p2 is None or abs(d_sca) < abs(d_p1p2))):
                return L[f"{sca_side}-1.9 BA-SCA junction"], "distal BA at SCA origin"
            return L["1.10 BA tip"], "distal BA (tip)"
        if d_aica is not None and abs(d_aica) <= P["ba_junction_mm"]:
            return L[f"{aica_side}-1.7 BA-AICA junction"], "mid BA at AICA origin"
        if P["ba_default"] == "tip":
            return L["1.10 BA tip"], "BA default=tip"
        if P["ba_default"] == "vaba":
            return L["1.5 VA-BA junction"], "BA default=VA-BA junction"
        return L["1.4 BA trunk"], "mid BA away from branches"

    if fam == "VA":
        d_pica = _delta(rec, "PICA", S, max_dist=P["va_junction_mm"])
        if af >= P["va_vaba_frac"]:
            return L["1.5 VA-BA junction"], "distal VA at basilar"
        if P["va_use_type"] != "off":
            ty = rec.get("aneurysm_type")
            if ty in ("dissecting", "fusiform"):
                return L[f"{S}-1.1 VA trunk"], f"VA, {ty}: segmental wall disease => trunk"
            if ty == "saccular" and P["va_use_type"] == "full":
                return L[f"{S}-1.3 VA-PICA junction"], "VA, saccular: branch-point aneurysm"
        if af >= P["va_pica_frac"] and d_pica is not None:
            return L[f"{S}-1.3 VA-PICA junction"], "distal VA at PICA origin"
        if P["va_default"] == "pica":
            return L[f"{S}-1.3 VA-PICA junction"], "VA default=PICA junction"
        return L[f"{S}-1.1 VA trunk"], "VA away from junctions"

    if fam == "PICA":
        return L[f"{S}-1.2 PICA trunk"], "parented on PICA"
    if fam == "SCA":
        return L[f"{S}-1.8 SCA trunk"], "parented on SCA"
    if fam == "AICA":
        return L[f"{S}-1.6 AICA trunk"], "parented on AICA"
    if fam == "P1P2":
        return L[f"{S}-2.1 P1P2"], "parented on P1P2"
    if fam == "P3P4":
        return L[f"{S}-2.2 P3P4"], "parented on P3P4"

    # ---- anterior circulation ----------------------------------------------------------------
    if fam == "Acom":
        return L["4.1 Acom complex"], "parented on Acom"

    if fam == "A1A2":
        # The A1A2 label spans A1 AND A2 as a single vessel, with the Acom attaching at the A1/A2
        # boundary. Two readings of that same boundary are available and the tuner picks between
        # them: the Acom delta is anatomically exact but the Acom is missing in 5 of 11 cases,
        # whereas arc_frac is always available and the boundary sits at a stable fraction (~0.44)
        # across patients -- the same population-reference trick that makes Tier C work.
        d_acom = _delta(rec, "Acom")
        if P["a1a2_use_acom"] and d_acom is not None:
            if abs(d_acom) <= P["a1a2_acom_mm"]:
                return L["4.1 Acom complex"], "A1A2 at Acom attachment"
            return (L[f"{S}-4.3 A2"], "A1A2 distal to Acom") if d_acom > 0 else \
                   (L[f"{S}-4.2 A1"], "A1A2 proximal to Acom")
        if af <= P["a1a2_a1_frac"]:
            return L[f"{S}-4.2 A1"], "proximal A1A2 = A1"
        if af <= P["a1a2_acom_frac"]:
            return L["4.1 Acom complex"], "A1A2 at the A1/A2 boundary"
        return L[f"{S}-4.3 A2"], "distal A1A2 = A2"

    if fam == "A3":
        # the A3/distal boundary is where the vessel model stopped labelling A3, not an
        # anatomical landmark, so this is the weakest decision in the set
        return L[f"{S}-4.4 A3"], "parented on A3"

    if fam in ("3rd-A2", "3rd-A3"):
        return ABSTAIN, "3rd-A2/A3 variant has no location class"

    return ABSTAIN, f"unhandled family {fam}"
