#FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
#FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip python3-dev python3-numpy 

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt install -y tzdata
RUN apt-get install -y cmake 
RUN apt-get install -y git
RUN apt-get install -y wget 
RUN apt-get install -y unzip 
RUN apt-get install -y yasm 
RUN apt-get install -y tmux 

RUN apt-get install -y bzip2 ca-certificates
RUN apt-get install -y pkg-config 
RUN apt-get install -y build-essential
RUN apt-get install -y tree
RUN apt-get install -y htop

# RUN apt-get install -y nvtop



     
RUN apt-get install -y \
     ffmpeg \
     libsm6 \
     libgl1 \
     libgl1-mesa-glx \
     libxext6 
RUN apt-get install -y \
     libfontconfig1 \
     libxrender1 \
     libswscale-dev \
     libtbb2 \
     libtbb-dev \
     libjpeg-dev 
     
RUN apt-get install -y \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libxine2-dev \
    libglew-dev \
    libtiff5-dev 
    
RUN apt-get install -y \
    zlib1g-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libpostproc-dev \
    libswscale-dev \
    libeigen3-dev \
    libgtk2.0-dev \
    libwebp-dev 

# Create conda env from scratch
ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh 

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Opencv complete installation
RUN apt-get update \
    && rm -rf /var/lib/apt/lists/*

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.8.0

RUN cd /opt/ &&\
    # Download and unzip OpenCV and opencv_contrib and delte zip files
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    # Create build folder and switch to it
    mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
    # Cmake configure
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=7.5,8.0,8.6 \
        -DCMAKE_BUILD_TYPE=RELEASE \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. &&\
    # Make
    make -j"$(nproc)" && \
    # Install to /usr/local/lib
    make install && \
    ldconfig &&\
    # Remove OpenCV sources and build folder
    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}
    
    
# # create conda env
RUN conda create -n Py3.9 python=3.9
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && echo "source activate Py3.9" > ~/.bashrc  
ENV PATH /opt/conda/envs/Py3.9/bin:$PATH

# # pip pkg on conda env
RUN pip3 install torch==1.13.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

RUN pip3 install opencv-python Pillow numpy scikit-image matplotlib imageio scipy plotly sympy
RUN pip3 install  onnx onnxruntime tqdm
RUN pip3 install mmcv==1.5.0 mmcv-full==1.5.0 mmdet==2.25.3 mmengine==0.8.4
RUN pip3 install cython submitit pycocotools 
RUN pip3 install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI&egg=pycocotools
RUN pip3 install git+https://github.com/cocodataset/panopticapi.git#egg=panopticapi
     
# RUN conda install -n Py3.9 pytorch=1.13.0 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
# RUN conda install -n Py3.9 -c fvcore -c iopath -c conda-forge fvcore iopath
# RUN conda install -n Py3.9 pytorch3d -c pytorch3d

