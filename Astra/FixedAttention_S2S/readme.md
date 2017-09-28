This is an implementation of Efficient Attention using a Fixed-Size Memory Representation - https://arxiv.org/abs/1707.00110

How to run:

1. Download ParlAI
2. In a terminal, do the following:

`cd ParlAI/parlai/agents`

`ln -s astra/RecurrentS2S_FixedAttention FixedAttnS2S`

3. Run with the following command:

`cd ParlAI/examples`

`python examples/train_model.py -m FixedAttnS2S -t dialog_babi:Task:1 -mf "/tmp/model" -vme 64 -vtim 30 -vp 100 -b 32`
