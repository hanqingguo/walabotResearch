import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from python.utils import *
from python.cnn_model import *


def train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25):
    pass


if __name__ == '__main__':
    device = torch.device("cuda:0")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    training_dir = os.path.join(os.path.dirname(current_dir), 'training')
    label = 'walk'
    idx = '1'
    video_format = '.avi'
    video_name = idx+video_format
    video_path = os.path.join(training_dir, label, video_name)

    setting = {'sequence_num': 10,
               'hidden_size': 3,
               'num_layers': 2,
               'num_directions': 1,
               'num_features': 100,
               'cut_frame': 10}
    """
    sequence_num: how many frame in the sequence
    hidden_size: hidden_size is hidden state dimension
    num_layers: upper layer of each lstm element
    num_directions: 2 is bidirection, 1 is only one direction
    """
    lstm = nn.LSTM(input_size=setting['num_features'], hidden_size=setting['hidden_size'],
                   num_layers=setting['num_layers'])

    h_0 = torch.randn(setting['num_layers'] * setting['num_directions'], 1, setting['hidden_size'])
    c_0 = torch.randn(setting['num_layers'] * setting['num_directions'], 1, setting['hidden_size'])
    inputs = getFeature(video_path, 'resnet18', setting['num_features'], setting['cut_frame'])

    inputs = inputs.unsqueeze(1)

    lstm = lstm.to(device)
    outputs, (h_n, c_n) = lstm(inputs)
    print("outputs.size is: {},"
          "h_n.size is: {},"
          "c_n.size is: {}".format(outputs.size(),
                                   h_n.size(),
                                   c_n.size()))
