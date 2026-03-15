FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    openssh-client \
    ca-certificates \
    curl \
    bash \
    libopenjp2-7-dev \
    libopenjp2-tools \
    openslide-tools \
    libgeos-dev \
    texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build

RUN uv python install 3.12

COPY pyproject.toml ./
COPY uv.lock* ./

RUN uv sync --python 3.12

RUN curl -fsSL https://opencode.ai/install | bash

ENV PATH="/root/.opencode/bin:/tmp/build/.venv/bin:${PATH}"

WORKDIR /workspace

CMD ["/bin/bash"]
