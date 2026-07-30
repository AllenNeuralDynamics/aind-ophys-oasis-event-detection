# hash:sha256:50c28457f4e252204dfff5c996a07c7c86ed488abac08dd74e83bdb0c204258f
FROM registry.codeocean.allenneuraldynamics.org/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV VERSION="4.0"

ENV REPO_URL=https://github.com/AllenNeuralDynamics/aind-ophys-oasis-event-detection

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN ["apt-get", "update"]
# build-essential: oasis-deconv 0.2.0 is published as an sdist only (no
# manylinux wheel) and Cython-compiles oasis_methods at install time.
RUN ["apt-get", "install", "-y", "vim", "build-essential"]

RUN pip install -U --no-cache-dir \
    h5py==3.9.0 \
    numpy==1.25.1 \
    oasis-deconv==0.2.0 \
    pydantic==2.8.2 \
    seaborn==0.12.2 \
    aind-data-schema==1.2.0 \
    aind-log-utils \
    aind-qcportal-schema
