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
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return self.pe[0, :x.size(1), :]

class EncoderTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=4):
        super(EncoderTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(2 * hidden_size, output_size)
        
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        pos_embedded = self.pos_encoder(input_seq)

        embedded = embedded + pos_embedded
        self_attention = self.transformer_encoder(embedded)
        self_attention = self_attention.to(self.device)
        
        h = self_attention[:, [0, -1], :].reshape([-1, 2 * self.hidden_size])
        
        ret = self.linear(h)
        return ret
