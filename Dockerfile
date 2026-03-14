FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# uv binaries
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
    less \
    vim \
    procps \
    libopenjp2-7-dev \
    libopenjp2-tools \
    openslide-tools \
    libgeos-dev \
    texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python runtime
RUN uv python install 3.12

# Copy only dependency descriptors for layer caching
COPY pyproject.toml ./
COPY uv.lock* ./

# Create env and install dependencies
RUN uv sync --python 3.12

# Install OpenCode
RUN curl -fsSL https://opencode.ai/install | bash

# Make opencode visible in shells
ENV PATH="/root/.opencode/bin:/workspace/.venv/bin:${PATH}"

CMD ["/bin/bash"]
