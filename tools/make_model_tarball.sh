#!/usr/bin/env bash
# Pack a model folder (layout: topaneu_infer/pipeline.py) into the tarball Grand Challenge expects.
#   bash tools/make_model_tarball.sh <model_dir> <out.tar.gz>
set -euo pipefail
MODEL=$1; OUT=$2
for p in ct/model.json mr/model.json mappings/vessel_mapping.json mappings/location_mapping.json params.json; do
  [ -f "$MODEL/$p" ] || { echo "missing $MODEL/$p" >&2; exit 1; }
done
# world-readable entries owned by root: the container runs as an unprivileged user
tar -C "$MODEL" --owner=0 --group=0 --numeric-owner --mode='u+rw,go+r,go-w,a+X' \
    -czf "$OUT" ct mr vessel mappings params.json
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
