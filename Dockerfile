FROM nvidia/cuda:10.0-cudnn7-devel

ENV CUDA_PATH /usr/local/cuda
ENV CUDA_ROOT /usr/local/cuda/bin
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64


RUN ldconfig
RUN apt-get -qq update
# libsm6 and libxext6 are needed for cv2
# Need gcc to build ASTRA toolbox
RUN apt-get update && apt-get install -y libxext6 \
                                         libsm6 \
                                         libxrender1 \
                                         libgl1-mesa-glx \
                                         build-essential \
                                         automake \
                                         libboost-all-dev \
                                         git \
                                         openssh-server \
                                         wget \
                                         nano \
                                         rsync && \
                                         rm -rf /var/lib/apt/lists/*

RUN ldconfig

RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH "$CUDA_ROOT:/root/miniconda3/bin:$PATH"

RUN conda install python=3.7
RUN conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing boost
RUN conda install pytorch=1.2.0 cudatoolkit=10.0 torchvision -c pytorch
RUN conda install scipy pandas cython matplotlib tqdm pillow scikit-learn scikit-image=0.14 -yq
RUN pip install opencv-python h5py -q
RUN pip install dominate visdom runstats -q
RUN pip install tb-nightly yacs -q
RUN pip install --upgrade pip
RUN pip install future packaging pytest coverage coveralls easydict tifffile demandimport simpleitk -q
RUN python --version
RUN pip list

WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/apex
WORKDIR apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .


# Create directories for input and output
RUN mkdir /manet && chmod 777 /manet

# ssh
RUN mkdir /var/run/sshd
RUN echo 'root:pwd' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

EXPOSE 22

WORKDIR /manet

# Provide an open entrypoint for the docker
ENTRYPOINT service ssh restart && $0 $@
