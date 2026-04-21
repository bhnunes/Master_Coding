FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG OPENSLIDE_VERSION=4.0.0

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV NO_UPDATE_NOTIFIER=true

# Pre-install curl and certificates to set up the NodeSource repository
RUN apt-get update && apt-get install -y curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y \
    nodejs \
    python3-pip \
    python3-dev \
    git \
    openssh-client \
    bash \
    build-essential \
    meson \
    ninja-build \
    pkg-config \
    libcairo2-dev \
    libglib2.0-dev \
    libgdk-pixbuf-2.0-dev \
    libsqlite3-dev \
    shared-mime-info \
    libopenjp2-7-dev \
    libopenjp2-tools \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libxml2-dev \
    zlib1g-dev \
    libgeos-dev \
    texlive-latex-extra \
    # Chromium/Puppeteer dependencies required for mermaid rendering in headless Linux
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/openslide-src

RUN curl -fsSL "https://github.com/openslide/openslide/releases/download/v${OPENSLIDE_VERSION}/openslide-${OPENSLIDE_VERSION}.tar.xz" \
    -o openslide.tar.xz \
    && tar -xf openslide.tar.xz --strip-components=1 \
    && meson setup builddir --buildtype=release \
    && meson compile -C builddir \
    && meson install -C builddir \
    && ldconfig \
    && rm -rf /tmp/openslide-src

WORKDIR /tmp/build

RUN uv python install 3.12

COPY pyproject.toml ./
COPY uv.lock* ./

RUN uv sync --python 3.12

RUN uv run python -c "import openslide; version = openslide.lowlevel.get_version(); print(f'OpenSlide native version: {version}'); major = int(version.split('.', 1)[0]); assert major >= 4, f'Expected native OpenSlide >= 4.0.0, got {version}'; cache = openslide.OpenSlideCache(1024 * 1024); print(f'OpenSlide cache check: {type(cache).__name__}')"

RUN curl -fsSL https://opencode.ai/install | bash

ENV PATH="/root/.opencode/bin:/tmp/build/.venv/bin:${PATH}"

WORKDIR /workspace

CMD ["/bin/bash"]