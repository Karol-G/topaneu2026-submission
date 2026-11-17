#!/usr/bin/env python3
"""Grand Challenge entry point, following the organizers' template
(github.com/Bangulli/TopAneu-26, templates/task{1,2}/main.py).

One image in, one output out, one container invocation per case. The input socket gives the
modality:

    /input/images/head-ct-angio/<uuid>.mha    -> CT
    /input/images/head-mr-angio/<uuid>.mha    -> MR

    Task 1: /output/detected-aneurysm-locations.json        list of location class ids
    Task 2: /output/images/aneurysm-segmentation/output.mha  uint8 mask, values 0-52

TOPANEU_TASK (baked in at build time) selects which output is written.
"""
import glob
import json
import os
import sys
import traceback
from pathlib import Path

import SimpleITK

from topaneu_infer import run_case

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
TASK = int(os.environ.get("TOPANEU_TASK", "2"))
MODEL_DIR = Path(os.environ.get("TOPANEU_MODEL_DIR", "/opt/ml/model"))
VESSEL_CODE = Path(os.environ.get("TOPANEU_VESSEL_CODE", "/opt/vessel"))


def run():
    slugs = tuple(sorted(sv["socket"]["slug"] for sv in load_json_file(INPUT_PATH / "inputs.json")))
    modality, folder = {
        ("head-ct-angiography",): ("ct", "head-ct-angio"),
        ("head-mr-angiography",): ("mr", "head-mr-angio"),
    }[slugs]
    image = load_image_file(INPUT_PATH / "images" / folder)
    segmentation, location_ids = run_case(image, modality, MODEL_DIR, VESSEL_CODE)
    if TASK == 1:
        write_json_file(OUTPUT_PATH / "detected-aneurysm-locations.json",
                        [int(v) for v in location_ids])
    else:
        out = OUTPUT_PATH / "images" / "aneurysm-segmentation"
        out.mkdir(parents=True, exist_ok=True)
        SimpleITK.WriteImage(segmentation, out / "output.mha", useCompression=True)
    return 0


def load_json_file(location):
    with open(location) as f:
        return json.loads(f.read())


def write_json_file(location, content):
    location.parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_file(location):
    files = (glob.glob(str(location / "*.tif")) + glob.glob(str(location / "*.tiff"))
             + glob.glob(str(location / "*.mha")))
    if not files:
        raise FileNotFoundError(f"no image in {location}")
    return SimpleITK.ReadImage(files[0])


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        raise SystemExit(1)
