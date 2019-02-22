from python.lstm_useCNN_feature import CNN,train_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from python.utils import *
from python.cnn_model import *

"""
日尼玛RNN学不到东西，用DNN试试看

Time: Feb.9.2019
"""

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(6*20,100)
        self.fc2 = nn.Linear(100,20)
        self.fc3 = nn.Linear(20,2)
    def forward(self, x):
        x = x.view(-1, 20*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # /home/hanqing/walabot_research/python
    data_dir = 'cut_dataset'
    setting = {
                'sequence_num': 10,
                'hidden_size': 100,
                'num_layers': 1,
                'num_directions': 1,
                'num_features': 512,
                'cut_frame': 6
              }

    classTable = {'walk': 0, 'still':1}
    model = DNN()
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    trained_model = train_model(model,criterion,optimizer, exp_lr_scheduler, current_dir, data_dir,setting,classTable,100)