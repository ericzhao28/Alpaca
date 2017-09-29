This is an implementation of Global Attention - https://arxiv.org/abs/1707.00110

Prerequisites:

1. Download ParlAI

2. In a terminal, do the following:

`cd ParlAI/parlai/agents`

`ln -s astra/RecurrentS2S_GlobalAttention GlobalAttnS2S`

Run:

`cd ParlAI/examples`

`python examples/train_model.py -m GlobalAttnS2S -t dialog_babi:Task:1 -mf "/tmp/model" -vme 64 -vtim 30 -vp 100 -b 32`
