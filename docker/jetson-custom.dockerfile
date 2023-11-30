# Custom dockerfile for building Ultralytics (YOLOv8) on Jetson TX2 dev kit running Jetpack 4.6.1
# Multistage building is used for installing the correct version of python

# Stage 1: Build Python 3.10.13 -----------------------------------------------------------------------------
FROM ubuntu:18.04 as python-builder
ARG PYTHON_VERSION=3.10.13

# Install build dependencies for python
RUN apt update && apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev

# Download and build Python
RUN curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j 8 && \
    make altinstall

# Stage 2: Build Your Application with the new Python version -----------------------------------------------
# Start FROM https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

# Copy updated Python from the builder stage
COPY --from=python-builder /usr/local /usr/local

# This resolves gpg error on 18.04 LTS seen in next command block
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 42D5A192B819C5DA
ENV LC_ALL=C.UTF-8

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages
# g++ required to build 'tflite_support' and 'lap' packages, libusb-1.0-0 required for 'tflite_support' package
RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0

# Create working directory
WORKDIR /usr/src/ultralytics

# Copy contents
# COPY . /usr/src/ultralytics  # git permission issues inside container
RUN git clone https://github.com/ultralytics/ultralytics -b main /usr/src/ultralytics
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt /usr/src/ultralytics/

# Remove opencv-python from requirements.txt as it conflicts with opencv-python installed in base image
RUN grep -v '^opencv-python' requirements.txt > tmp.txt && mv tmp.txt requirements.txt

# Set python3.10.4 as default
#RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 1
#RUN update-alternatives --config python
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1
#RUN update-alternatives --config python3

# Install pip packages manually for TensorRT compatibility https://github.com/NVIDIA/TensorRT/issues/2567
RUN python -m pip install --upgrade pip wheel
RUN pip install --no-cache tqdm matplotlib pyyaml psutil pandas onnx "numpy==1.23"
RUN pip install --no-cache -e .

# Install couple other packages manually
RUN apt install -y vim
RUN apt install -y liblzma-dev
RUN pip install backports.lzma

# Download openCV sourcecode, build, and install
WORKDIR /home
RUN apt update && apt install -y cmake wget unzip
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
RUN unzip opencv.zip
RUN mkdir -p build & cd build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON ../opencv-4.x
RUN make
RUN make install
RUN cd .. && rm opencv.zip

# Clone HARVEY.git to home directory
RUN git clone https://github.com/BSPLAZA/HARVEY.git
WORKDIR /home/HARVEY

# TODO: automatically modify lzma.py file with backport modification

# Set environment variables
ENV OMP_NUM_THREADS=1


# Usage Examples -------------------------------------------------------------------------------------------------------

# First pull required docker image
# sudo docker pull nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3


# Build
# sudo docker build -f jetson-custom.dockerfile -t ultralytics-jetson .

# Running built image
# sudo docker run -it --runtime nvidia --gpus all --privileged --publish-all ultralytics-jetson
