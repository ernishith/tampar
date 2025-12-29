FROM python:3.9-slim-bullseye


RUN apt-get -y update

# see https://serverfault.com/a/992421
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="America/New_York"

RUN apt-get install -y libopencv-dev python3-opencv git cmake graphviz
RUN apt-get install -y libmagickwand-dev

RUN mkdir -p /project
WORKDIR /project

# create a virtual environment to isolate pip installs
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

RUN pip install torch==1.10.2 torchvision==0.11.3 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
COPY requirements.txt /project/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# avoid exposing GPUs in the CPU image
ENV CUDA_VISIBLE_DEVICES=""
