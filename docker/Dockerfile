# syntax=docker/dockerfile:1

###############################################################################
### 1. Builder stage
###############################################################################
ARG CUDA_TAG=12.1.1-cudnn8-devel-ubuntu22.04

FROM nvcr.io/nvidia/cuda:${CUDA_TAG} AS builder

ENV DEBIAN_FRONTEND=noninteractive

ARG MAX_JOBS=4

ARG PYTORCH_CUDA=cu121
ARG PYTORCH_VERSION=2.1.2
ARG TORCHVISION_VERSION=0.16.2
ARG NUMPY_VERSION=1.26.4
ARG ME_COMMIT=4b628a7
ARG FAISS_COMMIT=e45ae24

ARG PIP_VERSION=25.1.1
ARG WHEEL_VERSION=0.45.1
ARG SETUPTOOLS_VERSION=69.0.3
ARG NINJA_VERSION=1.11.1.1

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        cmake \
        wget \
        swig \
        ninja-build \
        python3-dev \
        python3-pip \
        libopenblas-dev \
        libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# --Python base---------------------------------------------------------------
RUN python3 -m pip --no-cache-dir install \
        pip==${PIP_VERSION} \
        wheel==${WHEEL_VERSION} \
        setuptools==${SETUPTOOLS_VERSION} \
        ninja==${NINJA_VERSION} \
        numpy==${NUMPY_VERSION}

###############################################################################
### 1a. PyTorch wheel (including torchvision and dependencies)
###############################################################################
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheels \
        torch==${PYTORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}
# MinkowskiEngine require torch installation to build itself
RUN python3 -m pip install --no-cache-dir /wheels/torch*.whl

###############################################################################
### 1b. MinkowskiEngine wheel
###############################################################################
WORKDIR /build/mink
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CUDA_HOME=/usr/local/cuda-12.1
RUN git clone --recursive https://github.com/alexmelekhin/MinkowskiEngine.git \
        && cd MinkowskiEngine \
        && git checkout 6532dc3 \
        && python3 setup.py bdist_wheel \
                --force_cuda \
                --blas=openblas \
                --dist-dir /wheels

###############################################################################
### 1c. Faiss-GPU wheel
###############################################################################
# upgrade cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.5/cmake-3.26.5-linux-x86_64.sh && \
    mkdir /opt/cmake-3.26.5 && \
    bash cmake-3.26.5-linux-x86_64.sh --skip-license --prefix=/opt/cmake-3.26.5/ && \
    ln -s /opt/cmake-3.26.5/bin/* /usr/local/bin && \
    rm cmake-3.26.5-linux-x86_64.sh
WORKDIR /build/faiss
RUN git clone https://github.com/facebookresearch/faiss.git \
    && cd faiss \
    && git checkout c3b93749 \
    && cmake -B build . \
        -Wno-dev \
        -DFAISS_ENABLE_GPU=ON \
        -DFAISS_ENABLE_PYTHON=ON \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDAToolkit_ROOT=${CUDA_HOME} \
        -DCMAKE_CUDA_ARCHITECTURES="60;61;70;75;80;86" \
    && make -C build -j${MAX_JOBS} faiss \
    && make -C build -j${MAX_JOBS} swigfaiss \
    && cd build/faiss/python \
    && python3 setup.py bdist_wheel --dist-dir /wheels

###############################################################################
### 2. Dev/runtime stage
###############################################################################
FROM nvcr.io/nvidia/cuda:${CUDA_TAG} AS dev

ENV DEBIAN_FRONTEND=noninteractive

ARG INSTALL_ROS2=false
ENV ROS_DISTRO=humble

# — lightweight system packages for interactive work —
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        python-is-python3 \
        git \
        nano \
        vim \
        sudo \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

# WARNING: This allows sudo without password for all users in the sudo group
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# — create a user with the host's UID and GID —
ARG USER_NAME=docker_prism
ARG HOST_UID=1000
ARG HOST_GID=1000
ENV HOME=/home/${USER_NAME}
RUN groupadd --gid ${HOST_GID} ${USER_NAME} \
    && useradd --uid ${HOST_UID} \
               --gid ${HOST_GID} \
               --create-home \
               --shell /bin/bash \
               ${USER_NAME} \
    && usermod -aG sudo ${USER_NAME}

# — copy the compiled wheels and install them —
COPY --from=builder /wheels /tmp/wheels
RUN python3 -m pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# Install Open3D
RUN python3 -m pip install open3d

RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6

# Install OpenPlaceRecognition
RUN python3 -m pip install --upgrade setuptools \
    && python3 -m pip install --upgrade pip
RUN cd ${HOME} \
    && git clone --branch feat/toposlam https://github.com/OPR-Project/OpenPlaceRecognition \
    && cd OpenPlaceRecognition \
    && python3 -m pip install -e .

# - optional ROS2 Humble installation -
RUN if [ "$INSTALL_ROS2" = "true" ]; then \
        # setup repo
        apt-get update && apt-get install -y software-properties-common && add-apt-repository universe \
        && export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') \
        && curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" \
        && dpkg -i /tmp/ros2-apt-source.deb \
        # install base ROS2
        && apt-get update \
        && apt-get upgrade -y \
        && apt-get install -y --no-install-recommends \
            ros-${ROS_DISTRO}-ros-base \
            ros-dev-tools \
        && rm -rf /var/lib/apt/lists/*; \
    fi

RUN python3 -m pip install ros2-numpy scikit-image

# — working directory for convenience —
WORKDIR ${HOME}
USER ${USER_NAME}

RUN if [ "$INSTALL_ROS2" = "true" ]; then \
        # source ROS2 setup script
        echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ${HOME}/.bashrc; \
    fi

CMD ["/bin/bash"]
