FROM nvcr.io/nvidia/rapidsai/rapidsai:23.06-cuda11.8-runtime-ubuntu22.04-py3.10

USER root

# Set our locale to en_US.UTF-8.
ENV LANG en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8

RUN apt update && apt install -y software-properties-common
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
RUN apt-get update
RUN apt-get install -y libcudnn8
RUN apt-get install -y libcudnn8-dev
RUN apt-get install -y kmod
RUN apt-get update
RUN apt-get install -y nvidia-cuda-toolkit --fix-missing
RUN apt-get update && apt-get install -y ca-certificates python3-opencv git
RUN pip install gdown
RUN gdown 1-Fl0VggeQVqD7yGBPIFLpmbbNGlgknIw -O /var/ 
RUN dpkg -i /var/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
WORKDIR /var/cudnn-local-repo-ubuntu2204-8.9.7.29
RUN cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/cudnn-local-08A7D361-keyring.gpg /usr/share/keyrings/
RUN apt-get install -y libcudnn8=8.9.7.29-1+cuda12.2 --allow-downgrades && \
apt-get install -y libcudnn8-dev=8.9.7.29-1+cuda12.2 --allow-downgrades && \
apt-get install -y libcudnn8-samples=8.9.7.29-1+cuda12.2 --allow-downgrades && \
export LIBRARY_PATH=/usr/lib/cuda/nvvm/libdevice:$LIBRARY_PATH && \
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
WORKDIR /home/root
ARG CLONE_REPO="0"
RUN if [ "${CLONE_REPO}" = "2" ]; then git clone https://github.com/sohaibwaheedgithub/HR-VITON-CUSTOM.git HR-VITON-Custom; fi
WORKDIR /home/root/HR-VITON-Custom
RUN pip install -r requirements.txt