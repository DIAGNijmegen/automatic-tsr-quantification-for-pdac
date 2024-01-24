# Use the official nvidia/cuda base image
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
 
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Amsterdam
 
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
 
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server curl wget cmake libglib2.0-dev \
         gcc clang htop xz-utils ca-certificates \
        python3-pip python3.8 python3.8-dev libopencv-dev python3-opencv \
        libqt5concurrent5 libqt5core5a libqt5gui5 libqt5widgets5 \
        man  apt-transport-https sudo git subversion libunittest++2 \
        g++ meson ninja-build pv bzip2 zip unzip dcmtk libboost-all-dev \
        libgomp1 libjpeg-turbo8 libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev \
        libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev \
        libexpat1-dev liblzma-dev tk-dev gcovr libffi-dev uuid-dev \
        libgtk2.0-dev libgsf-1-dev libtiff5-dev libopenslide-dev \
        libgl1-mesa-glx libgirepository1.0-dev libexif-dev librsvg2-dev fftw3-dev orc-0.4-dev \
&& rm -rf /var/lib/apt/lists/*
 
 
 
# Import the NVIDIA GPG key
# RUN curl -s http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg
 
# # Configure APT to use the NVIDIA GPG keyring for signature verification
# RUN echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list > /dev/null
 
# Set the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
 
#Install jupyterlab
RUN pip3 install --upgrade pip setuptools
RUN pip3 install jupyterlab numpy==1.16.6
RUN pip3 install matplotlib==3.4
ARG CUDA_VERSION=11.1
ARG TORCH_VERSION=1.9.0
ARG TORCHVISION_VERSION=0.10.0
 
# Install PyTorch with GPU support
#RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0+cu111 -f https://download.pytorch.org/whl/cu111.html
RUN pip install torch==${TORCH_VERSION}+cu111 torchvision==${TORCHVISION_VERSION}+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# Clean up
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install git+https://github.com/qubvel/segmentation_models.pytorch
#User root
USER root
#print something 
RUN echo "Hello world"
ARG ASAP_URL=https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-1.9/ASAP-1.9-Linux-Ubuntu1804.deb
RUN : \
&& apt-get -y install curl \
&& curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/1.9/ASAP-1.9-Linux-Ubuntu1804.deb" \
&& dpkg --install ASAP-1.9-Linux-Ubuntu1804.deb || true \
&& apt-get clean \
&& rm ASAP-1.9-Linux-Ubuntu1804.deb
 

# Set the working directory
RUN pip3 install scikit-learn==1.0.2 scikit-image==0.18.0 albumentations==1.2.1 torchinfo wandb rdp
STOPSIGNAL SIGINT
EXPOSE 22 6006 8888
RUN useradd -m -s /bin/bash user && \
    echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R user:user /home/user/
COPY --chown=user pathology-common /home/user/source/pathology-common
COPY --chown=user pathology-fast-inference /home/user/source/pathology-fast-inference
COPY --chown=user models /home/user/source/models
COPY --chown=user code /home/user/source/code
# COPY pathology-common /home/user/pathology-common
# COPY pathology-fast-inference /home/user/pathology-fast-inference
# COPY code /home/user/code
# COPY models /home/user/models
USER user
ENV PYTHONPATH=/opt/ASAP/bin/:/home/user/source/pathology-common:/home/user/source/pathology-fast-inference:$PYTHONPATH
RUN export JUPYTER_PATH="${JUPYTER_PATH}:/opt/ASAP/bin:/home/user/source/pathology-common:/home/user/source/pathology-fast-inference"
RUN export PYTHONPATH="${PYTHONPATH}:/opt/ASAP/bin:/home/user/source/pathology-common:/home/user/source/pathology-fast-inference" # this does not work
# Command to run your application
COPY execute.sh /home/user/execute.sh
COPY start.sh /home/user/start.sh


ENV PYTHONPATH=/opt/ASAP/bin/:/home/user/source/pathology-common:/home/user/source/pathology-fast-inference:$PYTHONPATH
RUN export JUPYTER_PATH="${JUPYTER_PATH}:/opt/ASAP/bin:/home/user/source/pathology-common:/home/user/source/pathology-fast-inference"
RUN export PYTHONPATH="${PYTHONPATH}:/opt/ASAP/bin:/home/user/source/pathology-common:/home/user/source/pathology-fast-inference" # this does not work
WORKDIR /home/user
ENTRYPOINT /home/user/execute.sh
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
#CMD ["/bin/bash", "execute.sh"]