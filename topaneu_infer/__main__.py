"""Command line: one image in, the 52-class mask and the detected location ids out.

    python -m topaneu_infer -i case.nii.gz -m ct --model-dir model -o out/
"""
import argparse
import json
from pathlib import Path

import SimpleITK as sitk

from .pipeline import run_case

REPO_VESSEL_CODE = Path(__file__).resolve().parents[1] / "vessel"


def main():
    ap = argparse.ArgumentParser(prog="python -m topaneu_infer", description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--image", type=Path, required=True,
                    help="input image (.nii.gz, .mha, ... anything SimpleITK reads)")
    ap.add_argument("-m", "--modality", choices=("ct", "mr"), required=True)
    ap.add_argument("--model-dir", type=Path, required=True, help="extracted model tarball")
    ap.add_argument("-o", "--out-dir", type=Path, required=True)
    ap.add_argument("--vessel-code", type=Path, default=REPO_VESSEL_CODE,
                    help="the organizers' vessel model code (default: vessel/ in this repository)")
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="keep the binary aneurysm mask and the vessel mask in <out-dir>/work")
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    img = sitk.ReadImage(str(a.image))
    work = a.out_dir / "work" if a.keep_intermediates else None
    mask, ids = run_case(img, a.modality, a.model_dir, a.vessel_code, work_dir=work)
    sitk.WriteImage(mask, str(a.out_dir / "locations.nii.gz"), useCompression=True)
    (a.out_dir / "detected-aneurysm-locations.json").write_text(json.dumps([int(i) for i in ids]) + "\n")
    print(f"detected location classes: {ids}")


if __name__ == "__main__":
    main()
