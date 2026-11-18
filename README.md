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

## Repository layout

```
topaneu_infer/        the inference package
  segmenter.py          binary aneurysm model: preprocessing, sliding window, fold ensemble
  networks.py           network construction (Primus, ResEnc-L) from model.json
  resampling.py         image and logit resampling
  location/             centerline extraction and the location rules
  pipeline.py           the full chain for one case (run_case)
  __main__.py           command line
vessel/               the organizers' TA36 vessel model code, unchanged
container/            Dockerfile and Grand Challenge entry point
tools/                packing a model folder into the Grand Challenge tarball
```

## Reproduce

This repository reproduces the predictions of the submitted algorithm. It contains no training code.

**Requirements:** Linux, a CUDA GPU and about 20 GB of RAM. One case takes about 2–5 minutes.

### 1. Get the code and the weights

```bash
git clone https://github.com/Karol-G/topaneu2026-submission.git
cd topaneu2026-submission

# model tarball (3.0 GB), also at https://syncandshare.desy.de/index.php/s/fGnEo8NdzsEPZa4
curl -L -o topaneu2026_model.tar.gz \
    "https://syncandshare.desy.de/index.php/s/fGnEo8NdzsEPZa4/download?path=%2F&files=topaneu2026_model.tar.gz"
sha256sum topaneu2026_model.tar.gz
# ccedc235ea43bdc62693ff876773140f856368858da801efbb07c0ca92e2c615
mkdir model && tar -xzf topaneu2026_model.tar.gz -C model
```

The extracted folder:

```
model/
  ct/        model.json, fold_0.safetensors, fold_2.safetensors, fold_4.safetensors
  mr/        model.json, fold_0.safetensors, fold_2.safetensors, fold_3.safetensors
  vessel/    the organizers' TA36 vessel checkpoint (fold 4)
  mappings/  vessel_mapping.json, location_mapping.json
  params.json
```

`model.json` holds everything the binary model needs (architecture, patch size, target spacing,
axis order, resampling settings). The `.safetensors` files hold the network weights only.

### 2a. Run with Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.8.0+cu126 -r requirements.txt

python -m topaneu_infer -i case.nii.gz -m ct --model-dir model -o out/      # -m mr for MRA
```

Outputs in `out/`:

- `locations.nii.gz`: the Task 2 mask (uint8, 0 = background, 1–52 = location class), on the
  input image's grid.
- `detected-aneurysm-locations.json`: the Task 1 list of location class ids.

`--keep-intermediates` also keeps the binary aneurysm mask and the vessel mask in `out/work/`.

From Python:

```python
import SimpleITK as sitk
from topaneu_infer import run_case, Segmenter

img = sitk.ReadImage("case.nii.gz")
mask, location_ids = run_case(img, "ct", "model", "vessel")   # "ct" or "mr"

# only the binary aneurysm model
Segmenter("model/ct").predict_file("case.nii.gz", "aneurysm.nii.gz")
```

### 2b. Run the container

The container is the submitted algorithm. Grand Challenge runs it once per case, with the input
in `/input` and the extracted model tarball in `/opt/ml/model`. Task 1 and Task 2 use the same
code; a build argument selects which output is written.

```bash
docker build -f container/Dockerfile --build-arg TASK=2 -t topaneu-26-task2 .
docker build -f container/Dockerfile --build-arg TASK=1 -t topaneu-26-task1 .
```

To run one case locally, lay out the input as Grand Challenge does:

```
input/
  inputs.json                         [{"socket": {"slug": "head-ct-angiography"}}]
  images/head-ct-angio/case.mha       (MRA: slug head-mr-angiography, folder head-mr-angio)
```

```bash
mkdir -p output && chmod 777 output
docker run --rm --gpus all --shm-size 8g --network none \
    -v "$PWD/input:/input:ro" -v "$PWD/model:/opt/ml/model:ro" -v "$PWD/output:/output" \
    topaneu-26-task2
```

Task 2 writes `output/images/aneurysm-segmentation/output.mha`. Task 1 writes
`output/detected-aneurysm-locations.json`.

To upload the model to Grand Challenge, pack the extracted folder with
`bash tools/make_model_tarball.sh model model.tar.gz`. The image contains code only.

### 2c. Self-contained image

`container/Dockerfile.standalone` builds one image that also contains the model weights, so no
model tarball, mount or download is needed at run time. Build it from the repository root with the
extracted tarball in `model/`:

```bash
docker build -f container/Dockerfile.standalone -t topaneu-26-selfcontained .
docker run --rm --gpus all --shm-size 8g --network none \
    -v "$PWD/input:/input:ro" -v "$PWD/output:/output" topaneu-26-selfcontained
```

It writes both outputs (Task 1 json and Task 2 mask) by default; `-e TOPANEU_TASK=1` or
`-e TOPANEU_TASK=2` writes only one. `container/WEIGHTS.md` (in the image at
`/opt/ml/model/README.md`) describes the weights and their training data.

## References

- Isensee F. et al. *nnU-Net: a self-configuring method for deep learning-based biomedical image
  segmentation.* Nature Methods, 2021.
- Isensee F. et al. *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image
  Segmentation.* MICCAI, 2024.
- Wald T., Roy S., Isensee F. et al. *Primus: Enforcing Attention Usage for 3D Medical Image
  Segmentation.* arXiv:2503.01835, 2025.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
