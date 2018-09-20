import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class INIT_LSTM():
    def __init__(self, sequence_num, hidden_size, num_layers, num_directions):
        self.sequence_num = sequence_num
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions


lstm = nn.LSTM(input_size=sequence_num, hidden_size=hidden_size, num_layers=num_layers)

h_0 = torch.randn(num_layers*num_directions)