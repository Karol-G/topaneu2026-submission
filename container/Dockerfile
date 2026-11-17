# TopAneu 2026 submission container.
#
#   docker build -f container/Dockerfile --build-arg TASK=1 -t topaneu-26-task1 .
#   docker build -f container/Dockerfile --build-arg TASK=2 -t topaneu-26-task2 .
#
# Build context is the repository root. Model weights are not part of the image: Grand Challenge
# extracts the model tarball to /opt/ml/model (layout: see topaneu_infer/pipeline.py).

FROM --platform=linux/amd64 pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    nnUNet_n_proc_DA=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --no-cache-dir --no-color -r /opt/app/requirements.txt

# the organizers' vessel model code, run as a subprocess with its own import path
COPY --chown=user:user vessel /opt/vessel

COPY --chown=user:user topaneu_infer /opt/app/topaneu_infer
COPY --chown=user:user container/main.py /opt/app/

ARG TASK=2
ENV TOPANEU_TASK=${TASK} \
    TOPANEU_MODEL_DIR=/opt/ml/model \
    TOPANEU_VESSEL_CODE=/opt/vessel \
    PYTHONPATH=/opt/app

USER user
ENTRYPOINT ["python", "main.py"]
