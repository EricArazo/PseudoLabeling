FROM nvidia/cuda:8.0-cudnn5-devel

RUN apt update -y && apt install -y \
python3 \
python3-pip

RUN pip3 install \
torchvision==0.2.1 \
matplotlib==3.0.1 \
scikit-learn==0.20.0 \
tqdm==4.28.1 \
numpy==1.15.3 \
torchcontrib==0.0.2