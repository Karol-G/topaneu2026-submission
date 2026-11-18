# Model weights in this image

This image contains **one set of weights**, the one used for the final TopAneu 2026 submission.

**Training data: TopAneu 2026 training data only.** No external training data (images or labels)
was used, so there is no second "with external data" variant.

| Folder | Model | Training |
|---|---|---|
| `ct/` | Primus, binary aneurysm segmentation, folds 0, 2, 4 | TopAneu 2026 CTA training cases. Initialised from a publicly available self-supervised checkpoint (pretrained weights, no labels), then fine-tuned. |
| `mr/` | nnU-Net ResEnc-L, binary aneurysm segmentation, folds 0, 2, 3 | TopAneu 2026 MRA training cases, trained from scratch. |
| `vessel/` | the organizers' 36-class vessel model (fold 4) | unchanged, as released by the organizers |
| `params.json`, `mappings/` | location rule thresholds and label mappings | fitted on the TopAneu 2026 training cases |
