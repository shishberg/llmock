FROM mcr.microsoft.com/devcontainers/go:2-1.25-trixie
# FROM golang:1.25-trixie

USER root
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y nodejs npm libicu-dev \
    && npm install -g npm@latest @vue/cli @anthropic-ai/claude-code@latest

# Upgrade Go to 1.25.7
RUN ARCH=$(dpkg --print-architecture) \
    && curl -fsSL "https://go.dev/dl/go1.25.7.linux-${ARCH}.tar.gz" \
    | tar -C /usr/local -xz

# Install beads task tracker (as vscode user so it goes to ~/go/bin)
USER vscode
ENV PATH=$PATH:/home/vscode/.local/bin:/home/vscode/go/bin
# RUN go install github.com/steveyegge/beads/cmd/bd@v0.49.3
RUN curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash

ENV BEADS_NO_DAEMON=1
ENV TZ="Australia/Sydney"
