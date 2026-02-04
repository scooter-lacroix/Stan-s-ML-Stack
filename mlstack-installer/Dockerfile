# =============================================================================
# Stan's ML Stack - Rusty-Stack Installer
# Multi-stage Dockerfile for optimized Rust builds
# =============================================================================

# Stage 1: Builder
FROM rust:1.78-slim AS builder

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY . .

WORKDIR /build/rusty-stack

RUN cargo build --release

# Stage 2: Runtime
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ca-certificates \
    bash \
    sudo \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/rusty-stack/target/release/Rusty-Stack /usr/local/bin/Rusty-Stack
COPY --from=builder /build/scripts /app/scripts

RUN chmod +x /usr/local/bin/Rusty-Stack /app/scripts/*.sh /app/scripts/*.py

ENTRYPOINT ["/usr/local/bin/Rusty-Stack"]
