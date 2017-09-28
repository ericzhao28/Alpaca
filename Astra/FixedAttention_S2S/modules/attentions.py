import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LocalAttention(nn.Module):
    def __init__(self, ):
        super(LocalAttention, self).__init__()


class Memory(nn.Module):
    def __init__(self, mem_size, input_size):
        """
        :param attn_size: Number of attention vectors
        :param input_size: hidden_size + num_layers*num_direrctions
        """
        super(Memory, self).__init__()
        self.mem_size = mem_size
        self.input_size = input_size

        self.linear = nn.Linear(input_size, self.mem_size)
        self.softmax = nn.Softmax()

    def forward(self, encoder_state):
        """
        :param encoder_state: Flattened final encoder state
        :return:
        """
        lin_out = self.linear(encoder_state)
        alpha = self.softmax(lin_out)
        print(alpha.size())
        print(lin_out.size())
        memory = torch.mm(alpha, lin_out.t())

        return memory


class AttentionOverMemory(nn.Module):
    def __init__(self, mem_size, input_size):
        """
        :param mem_size:
        :param input_size:
        """
        super(AttentionOverMemory, self).__init__()
        self.mem_size = mem_size
        self.input_size = input_size

        self.linear = nn.Linear(input_size, mem_size)
        self.softmax = nn.Softmax()

    def forward(self, decoder_state, memory):
        """
        :param decoder_state:
        :return:
        """
        lin_out = self.linear(decoder_state)
        beta = self.softmax(lin_out)
        attn_vector = torch.bmm(beta, memory)

        return attn_vector






