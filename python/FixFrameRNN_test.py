import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from python.utils import *
from python.cnn_model import *
import pandas as pd

class CNN(nn.Module):
    def __init__(self, ori_model):
        super(CNN,self).__init__()
        self.conv1 = ori_model.conv1
        self.bn1 = ori_model.bn1
        self.relu = ori_model.relu
        self.maxpool = ori_model.maxpool
        self.layer1 = nn.Sequential(*list(ori_model.layer1.children()))
        self.layer2 = nn.Sequential(*list(ori_model.layer2.children()))
        self.layer3 = nn.Sequential(*list(ori_model.layer3.children()))
        self.layer4 = nn.Sequential(*list(ori_model.layer4.children()))
        self.maxpool1 = nn.MaxPool2d(kernel_size=7, stride=1, padding=0)
        #self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpool1(x)
        #x = self.avgpool(x)
        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(RNN, self).__init__()
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
        #print("out.size() is {}".format(out.size()))
        out = out[-1, :, :]
        out = F.softmax(out, dim=1)


        return out

def testVideo(video_path, rnn):
    ori_model = models.resnet18(pretrained=True)
    CNN_model = CNN(ori_model)
    CNN_model = CNN_model.to(device)
    inputs = getFeature2(video_path,CNN_model, 3)
    print(inputs)
    inputs = inputs.unsqueeze(1)
    output = rnn(inputs)

    return output


if __name__ == '__main__':
    device = torch.device("cuda:0")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    #current_dir = / home / hanqing / walabot_Research / walabotResearch / python
    # In video_loader function, get dirname of current_dir to /home/hanqing/walabot_Research/ walabotResearch/
    data_dir = 'cut_dataset'



    setting = {'cnn_model': 'resnet18',
               'sequence_num': 8,
               'hidden_size': 6,
               'num_layers': 1,
               'num_directions': 1,
               'num_features': 512,
               'cut_frame': 3}
    """
    sequence_num: how many frame in the sequence
    hidden_size: hidden_size is hidden state dimension
    num_layers: upper layer of each lstm element
    num_directions: 2 is bidirection, 1 is only one direction
    """
    #classTable = {'walk':0, 'sit-to-stand':1, 'stand-to-sit':2, 'fall_down':3, 'jump':4}

    classTable = {'jump': 0, 'walk': 1}


    model = RNN(input_size=setting['num_features'], hidden_size=setting['hidden_size'],
                num_layers=setting['num_layers'], num_class=len(classTable))
    model = model.to(device)

    saved_path = os.path.join(current_dir, 'activity_best_para')
    model.load_state_dict(torch.load(saved_path))

    out = testVideo('/home/hanqing/walabot_research/cut_dataset/walk/19_cut.avi', model)
    print(out)
