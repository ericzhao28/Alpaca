import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .attentions import AttentionOverMemory

class MemoryDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mem_size, num_layers=1, batch_first=True, decoder_type='gru'):
        super(MemoryDecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        if decoder_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=batch_first)
        elif decoder_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=batch_first)

        self.attn = AttentionOverMemory(mem_size=mem_size, input_size=input_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, x, hidden, memory):

        attn_applied = self.attn.forward(hidden, memory=memory)
        output = torch.add(x, attn_applied)

        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)

        scores = F.log_softmax(self.out(output[0]))
        return scores
