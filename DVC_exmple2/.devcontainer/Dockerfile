FROM mcr.microsoft.com/devcontainers/python:3.11



#COPY --from=jdxcode/mise /usr/local/bin/mise /usr/local/bin/


# System packages only

RUN apt-get update && apt-get install -y \
    bash curl git ca-certificates build-essential sudo \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://mise.run | MISE_INSTALL_PATH=/usr/local/bin/mise sh 
 

RUN echo 'eval "$(mise activate bash)"' >> /home/vscode/.bashrc
