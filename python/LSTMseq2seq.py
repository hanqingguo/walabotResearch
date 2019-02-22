from python.lstm_useCNN_feature import CNN,train_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from python.utils import *
from python.cnn_model import *

"""
每个视频的每帧都拿结果
每帧都学习
Time: Feb.22.2019
"""

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.classifier = nn.Linear(hidden_size, num_class)

    def forward(self, inputs):
        h_0 = torch.randn(self.num_layers, 1, self.hidden_size)
        c_0 = torch.randn(self.num_layers, 1, self.hidden_size)
        h_0 = nn.init.orthogonal_(h_0)
        c_0 = nn.init.orthogonal_(c_0)
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        out, _ = self.lstm(inputs,(h_0,c_0))
        out = self.classifier(out)
        out = out[:, :, :]
        out = F.log_softmax(out, dim=1)

        return out

if __name__ == "__main__":
    device = torch.device("cuda:0")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # /home/hanqing/walabot_research/python
    data_dir = 'cut_dataset'
    setting = {
                'sequence_num': 6,
                'hidden_size': 256,
                'num_layers': 1,
                'num_directions': 1,
                'num_features': 64,
                'cut_frame': 6
              }
    classTable = {'walk': 0, 'still':1}
    model = LSTM(setting['num_features'], setting['hidden_size'], setting['num_layers'],len(classTable))
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    trained_model = train_model(model,criterion,optimizer, exp_lr_scheduler, current_dir, data_dir,setting,classTable,100)