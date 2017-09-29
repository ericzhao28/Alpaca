# Astra
Deep Chatbot Platform

### Overview
Deep learning chat platform offering pretrained seq2seq and retreival-based models for predicting responses in natural language discoure. Models are ensembled with reinforcement learning techniques leveraged for model selection. Model training and deployment setups leverage distributed computing capabilities for 24/7 training and minimal downtime.

Versions:
* Docker-CE Version 17.06.0-ce-mac19
* Python 3.6.1

Python dependencies:
* ParlAI 0.1.0

### Technologies

##### Models
Models are written in PyTorch and Python 3+, with standard scientific computing dependencies like Numpy.
Existing models:
* Reinforcement-learning ensemble model-selection by Eric
* Retreival-based model by Monireh
* Seq2seq with fixed attention by Srikrishna
* Seq2seq with global attention by Srikrishna
* Vanilla Seq2seq by Srikrishna
* Convultional seq2seq by Srikrishna

##### Data pipeline
ParlAI is used as a general architecture and to serve data to models.
Datasets used:
* dialog_babi
* insurance Q&A

##### Deployment
Model deployment and the entire training pipeline are deployed as Docker Compose applications. The Compose app is deployed to Docker Swarm, which automatically partitions Ec2 resources. Deployed models are served through a synchronous API (solved by multiple simultaneous APIs in the swarm), and accessed through a GraphQL layer.

### Installation
```
docker-compose up --build
bash access_cluster.sh
cd /service/ParlAI/parlai/agents
ln -sf ../../../Astra/RecurrentS2S_Vanilla ./S2S
cd /service/ParlAI/
pip install -e .
```

### Usage
Go:
```
cd /service/ParlAI/
python3.6 examples/train_model.py -m S2S -t dialog_babi:Task:1 -mf "/tmp/model" -vme 64 -vtim 30 -vp 100 -b 32
```

### Maintenance 
Maintenance is intended. Got changes? Make a pull request.

### Contributors
Contributors are listed under contributors.txt

### Research papers on modelling human conversational discourse
Most are as suggested by Srikrishna.

##### Ensemble
Reinforcement learning, Markov decision sequence (MILABOT): https://arxiv.org/pdf/1709.02349.pdf

##### Retrival-based:
VHRED: https://arxiv.org/abs/1605.06069
Skip-thought: https://arxiv.org/abs/1506.06726

##### Seq2seq
Basic paper: https://arxiv.org/pdf/1506.05869v2.pdf
Tutorial:  https://docs.google.com/presentation/d/1quIMxEEPEf5EkRHc2USQaoJRC4QNX6_KomdZTBMBWjk/edit?usp=sharing
Convolutional Seq2Seq: https://arxiv.org/abs/1705.03122 
New objective function: https://arxiv.org/pdf/1510.03055v2.pdf
Multi task learning: https://arxiv.org/pdf/1702.01932.pdf

##### Seq2seq w/ Attention:
Original attention paper (for language): https://arxiv.org/pdf/1409.0473.pdf 
Improved attention paper (includes local attention):  https://arxiv.org/pdf/1508.04025.pdf 
Memory based attention: https://arxiv.org/pdf/1707.00110.pdf 
Using attention to replace conv. + recurrent in seq2seq: http://export.arxiv.org/pdf/1706.03762 

### License
Copyright 2017 Hive Data and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

