# Use NVIDIA/CUDA base image with Ubuntu 20.04
FROM  nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive

# Update APT package list and install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    wget \
    curl \
    gnupg \
    cuda-command-line-tools-11-8 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Install vscode server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Install Miniconda package manager
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
  bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
  rm /tmp/Miniconda3-latest-Linux-x86_64.sh && \
  ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
  echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
  echo "conda activate base" >> ~/.bashrc


# Create and activate the 'vida' Conda environment
RUN conda create --name vida python=3.8 && \
    conda update -n base conda && \
    echo "conda activate vida" >> ~/.bashrc  

# Install Conda and pip packages
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate vida && \
    # conda install -y yaml && \
    pip install PyYAML && \
    pip install -U scikit-learn && \
    pip install numpy==1.22.4 matplotlib==3.5.1 networkx==3.1 tensorboard plotly && \
    conda install -y ipykernel && \
    pip install phate && \
    pip install nbconvert && \
    conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia && \
    python -m ipykernel install --name=vida

# Set an environment variable for VIDA path
ENV vida_path="/app/vida"

# Copy run_vida.sh to the VIDA path
COPY . $vida_path/

# Define entrypoint or command
CMD ["bin/bash"]

