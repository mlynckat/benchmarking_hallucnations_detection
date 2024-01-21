FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt update
RUN apt install -y git git-lfs nano python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


WORKDIR /usr/src/app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 22