FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# 1. Install uv binary directly
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Add this to your Dockerfile
ENV CUDA_VISIBLE_DEVICES=""

# 2. Install System Dependencies (OpenSlide, LaTeX)
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git libopenjp2-7-dev \
    libopenjp2-tools openslide-tools texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 3. Copy project files and sync environment
# Note: We use --system to install directly into the container
COPY pyproject.toml .
RUN uv sync --system --no-dev  # Installs core deps
RUN uv sync --system           # Installs dev tools (ruff, mypy, pytest)

# Set up the OpenCode Agent auto-install
RUN curl -fsSL https://opencode.ai/install | bash