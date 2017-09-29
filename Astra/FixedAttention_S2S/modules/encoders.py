import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, encoder_type='gru'):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if encoder_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=batch_first)
        elif encoder_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=batch_first)

    def forward(self, x, hidden):
        # input_length = x.size()[0]
        # encoder_out = []
        # encoder_hidden = []

        for i in range(self.num_layers):
            x, hidden = self.rnn(x, hidden)

        # encoder_out = Variable(torch.FloatTensor(encoder_out))
        # encoder_hidden = Variable(torch.FloatTensor(encoder_hidden))
        return x, hidden


