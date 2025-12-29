FROM python:3.9-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="America/New_York"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    gcc build-essential \
    python3-dev \
    libopencv-dev python3-opencv \
    git cmake graphviz \
    libmagickwand-dev \
  && rm -rf /var/lib/apt/lists/*


WORKDIR /project

RUN python -m pip install --upgrade pip
