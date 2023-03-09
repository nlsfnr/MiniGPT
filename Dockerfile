FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y build-essential \
                       libssl-dev \
                       libffi-dev \
                       python3-dev \
                       python3-pip \
                       git \
                       vim

ENV PYTHONDONTWRITEBYTECODE=1
RUN git config --global --add safe.directory /workdir
WORKDIR /workdir
ADD ./requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python3 -m pip install -r requirements.txt
