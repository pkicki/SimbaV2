FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-12.2
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# packages
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      rsync \
      tree \
      curl \
      wget \
      unzip \
      htop \
      tmux \
      xvfb \
      patchelf \
      ca-certificates \
      bash-completion \
      libjpeg-dev \
      libpng-dev \
      ffmpeg \
      cmake \
      swig \
      libssl-dev \
      libcurl4-openssl-dev \
      libopenmpi-dev \
      python3-dev \
      zlib1g-dev \
      qtbase5-dev \
      qtdeclarative5-dev \
      libglib2.0-0 \
      libglu1-mesa-dev \
      libgl1-mesa-dev \
      libvulkan1 \
      libgl1-mesa-glx \
      libosmesa6 \
      libosmesa6-dev \
      libglew-dev \
      mesa-utils && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /root/.ssh

# python
RUN apt-get -y update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 python3.10-distutils python3.10-venv

## kubernetes authorization
# kubernetes needs a numeric user apparently
# Ensure the user has write permissions
RUN useradd --create-home \
    --shell /bin/bash \
    --base-dir /home \
    --groups dialout,audio,video,plugdev \
    --uid 1000 \
    user
USER root
WORKDIR /home/user
RUN chown -R user:user /home/user && \
    chmod -R u+rwx /home/user
USER 1000

# Install packages
COPY deps/requirements.txt /home/user/requirements.txt
RUN python3.10 -m venv /home/user/venv && \
    . /home/user/venv/bin/activate && \
    pip install -r requirements.txt && \
    pip install -U "jax[cuda12]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
ENV VIRTUAL_ENV=/home/user/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false 

# install mujoco 2.1.0, humanoid-bench and myosuite
ENV MUJOCO_GL egl
ENV LD_LIBRARY_PATH /home/user/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
RUN wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
     tar -xzf mujoco210-linux-x86_64.tar.gz && \
     rm mujoco210-linux-x86_64.tar.gz && \
     mkdir /home/user/.mujoco && \
     mv mujoco210 /home/user/.mujoco/mujoco210 && \
     find /home/user/.mujoco -exec chown user:user {} \; && \
     python -c "import mujoco_py" && \
     git clone https://github.com/joonleesky/humanoid-bench /home/user/humanoid-bench && \
     pip install -e /home/user/humanoid-bench && \
     git clone --recursive https://github.com/joonleesky/myosuite /home/user/myosuite && \
     pip install -e /home/user/myosuite

#############################
# Sony-AI
# # set authorization
RUN git clone https://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@github.com/dojeon-ai/SimbaV2.git /home/user/scale_rl
RUN git config --global --add safe.directory /home/user/scale_rl
ENV WANDB_API_KEY=96022b49a4e5c639895ba1e229022e087f79c84a

## Install dart dependencies
RUN --mount=type=secret,id=pip_extra_index_url,uid=1000 \
    PIP_EXTRA_INDEX_URL=$(cat /run/secrets/pip_extra_index_url); \
    pip install --disable-pip-version-check --no-cache-dir --extra-index-url ${PIP_EXTRA_INDEX_URL} dart-wrapper==0.1.3 dart-client==0.5.23
RUN pip install mypy-boto3-s3==1.28.55 boto3==1.28.24 boto3-stubs==1.28.74 awscli==1.29.24 protobuf==3.20.1
