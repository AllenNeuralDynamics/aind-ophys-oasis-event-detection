# hash:sha256:3c09b543c48e44c5101c6214c0ea6b12b9be7764a1be670f4e8152f31acd1e1d
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.4.4-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PIPELINE_URL="https://codeocean.allenneuraldynamics.org/capsule/6127754/tree"
ENV PIPELINE_VERSION="1.0"
ENV VERSION="6.0"

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]

RUN pip install -U --no-cache-dir \
    h5py==3.9.0 \
    numpy==1.25.1 \
    oasis-deconv==0.2.0 \
    pydantic==2.8.2 \
    seaborn==0.12.2 \
    aind-data-schema==0.22.0
