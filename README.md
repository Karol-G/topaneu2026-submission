# TopAneu 2026 — inference

Inference code of the ACVL / Helmholtz Imaging / DKFZ submission to the
[TopAneu 2026](https://topaneu-26.grand-challenge.org) challenge (MICCAI 2026): detection of
intracranial aneurysms in CTA and MRA and their assignment to one of 52 anatomical location classes.

- **Task 1** outputs the list of location classes in which an aneurysm was detected.
- **Task 2** outputs a voxel mask with values 0–52.

Both tasks run the same pipeline and differ only in which output is written.

## Method

```
image ──► binary aneurysm segmentation (CT or MR model, 3-fold ensemble) ──► aneurysm mask ─┐
                                                                                            ├─► location rules ─► 52 classes
image ──► 36-class vessel segmentation (organizers' TA36 model) ──────────► vessel mask ───┘
```

1. **Aneurysm segmentation.** No network predicts the 52 location classes. A binary
   aneurysm-vs-background model runs per modality, with nnU-Net preprocessing and inference
   (sliding window, Gaussian weighting, logits averaged over the folds):
   - **CT:** a Primus transformer (Wald et al., 2025), initialised from a publicly available
     self-supervised checkpoint and fine-tuned.
   - **MR:** nnU-Net ResEnc-L (Isensee et al., 2024), trained from scratch.
2. **Vessel segmentation.** The organizers' published 36-class vessel model (one model, fold 4),
   in `vessel/`.
3. **Location rules** (`topaneu_infer/location/`). For each aneurysm component:
   - the nearest vessel label becomes its parent vessel;
   - the parent's centerline is extracted and oriented proximal → distal using an adjacent
     vessel of known anatomical relation;
   - the aneurysm and every neighbouring branch origin are projected onto that centerline.

   Rules per vessel family then turn the arc position and the branch offsets into a location
   class. Their thresholds (`params.json`) were fitted on the training data.

Inference uses no test-time augmentation.

## Model folder

```
<model_dir>/
  ct/  model.json, fold_0.safetensors, fold_2.safetensors, fold_4.safetensors
  mr/  model.json, fold_0.safetensors, fold_2.safetensors, fold_3.safetensors
  vessel/Dataset572_TopAneu_Vessel_36fgCls_wLRSwap/resEncM_.../   (organizers' checkpoint)
  mappings/  vessel_mapping.json, location_mapping.json
  params.json
```

The trained weights are available on request.

## Usage

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.8.0+cu126 -r requirements.txt
```

```python
import SimpleITK as sitk
from topaneu_infer import run_case

img = sitk.ReadImage("case.nii.gz")
mask, location_ids = run_case(img, "ct", "/path/to/model_dir", "vessel")   # "ct" or "mr"
sitk.WriteImage(mask, "locations.nii.gz")
```

Only the binary aneurysm model:

```python
from topaneu_infer import Segmenter

Segmenter("/path/to/model_dir/ct").predict_file("case_0000.nii.gz", "aneurysm.nii.gz")
```

### Container (Grand Challenge)

```bash
docker build -f container/Dockerfile --build-arg TASK=1 -t topaneu-26-task1 .
docker build -f container/Dockerfile --build-arg TASK=2 -t topaneu-26-task2 .
bash tools/make_model_tarball.sh /path/to/model_dir model.tar.gz
```

The image contains code only. Grand Challenge extracts the model tarball to `/opt/ml/model`.

## References

- Isensee F. et al. *nnU-Net: a self-configuring method for deep learning-based biomedical image
  segmentation.* Nature Methods, 2021.
- Isensee F. et al. *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image
  Segmentation.* MICCAI, 2024.
- Wald T., Roy S., Isensee F. et al. *Primus: Enforcing Attention Usage for 3D Medical Image
  Segmentation.* arXiv:2503.01835, 2025.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
