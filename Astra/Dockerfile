From ubuntu:17.10

RUN apt update
RUN apt install build-essential cmake make wget git -y
RUN apt install python3.6 python3.6-dev -y
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py

WORKDIR /service

RUN pip3 install http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl 
RUN pip3 install torchvision

RUN git clone https://github.com/facebookresearch/ParlAI.git /service/ParlAI
RUN cd /service/ParlAI; pip3 install ./requirements.txt; echo "" > README.md; python3.6 setup.py develop

COPY ./Astra/Vanilla_S2S /service/ParlAI/parlai/agents/S2S
COPY ./Astra/GlobalAttention_S2S /service/ParlAI/parlai/agents/GlobalAttnS2S
COPY ./Astra/FixedAttention_S2S /service/ParlAI/parlai/agents/FixedAttnS2S

# RUN cd /service/ParlAI; python3.6 ./examples/train_model.py -m S2S -t dialog_babi:Task:1 -mf "/tmp/model" -vme 64 -vtim 30 -vp 100 -b 32
# RUN cd /service/ParlAI; python3.6 ./examples/train_model.py -m GlobalAttnS2S -t dialog_babi:Task:1 -mf "/tmp/model" -vme 64 -vtim 30 -vp 100 -b 32
# RUN cd /service/ParlAI; python3.6 ./examples/train_model.py -m FixedAttnS2S -t dialog_babi:Task:1 -mf "/tmp/model" -vme 64 -vtim 30 -vp 100 -b 32
