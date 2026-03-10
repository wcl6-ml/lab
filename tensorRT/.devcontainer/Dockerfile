FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update && apt-get install -y \
    tree vim bash-completion sudo curl git \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode
RUN useradd -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN curl https://mise.run | MISE_INSTALL_PATH=/usr/local/bin/mise sh

USER $USERNAME
WORKDIR /workspace
