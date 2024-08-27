import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(num_layers * 2 * hidden_size, output_size)
        
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        _, (h, _) = self.lstm(embedded)
        #h = h.to(self.device)
        
        h = h.transpose(0, 1).reshape(-1, self.num_layers * 2 * self.hidden_size)
        ret = self.linear(h)
        return ret


class EncoderCNN(nn.Module):
    def __init__(self, input_size, window_size, output_size, input_len=1000):
        super(EncoderCNN, self).__init__()
        self.input_size = input_size
        self.input_len = input_len
        assert window_size % 2 == 0, "window_size must be even number"

        self.conv1 = nn.Conv1d(input_size, 5, window_size, padding=window_size//2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(5, 10, window_size, padding=window_size//2)
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv1d(10, 20, window_size, padding=window_size//2)
        self.linear = nn.Linear(input_len * 20, output_size)
    def forward(self, input_seq):
        batch_size = input_seq.size()[0]
        out = F.one_hot(input_seq.long(), self.input_size).float()
        out = torch.transpose(out, 1, 2)

        out = self.conv1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        out = out[:, :, :self.input_len]

        out = self.conv2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = out[:, :, :self.input_len]

        out = self.conv3(out)
        out = out[:, :, :self.input_len]
        out = self.linear(out.reshape(batch_size, self.input_len * 20))
        return out
    
