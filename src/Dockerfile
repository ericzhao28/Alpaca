From ubuntu:17.10

RUN apt update
RUN apt install build-essential cmake make wget git -y
RUN apt install python3.6 python3.6-dev -y
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py

WORKDIR /service

RUN pip3 install http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl 
RUN pip3 install torchvision
