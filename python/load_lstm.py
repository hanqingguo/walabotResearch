import torch
import torch.nn as nn
import cv2
from python.lstm_useCNN_feature import RNN, CNN
from python.cnn_model import *



device = torch.device("cuda:0")

ori_model = models.resnet18(pretrained=True)
CNN_model = CNN(ori_model)
CNN_model = CNN_model.to(device)

setting = {
    'sequence_num': 8,
    'hidden_size': 100,
    'num_layers': 1,
    'num_directions': 1,
    'num_features': 64,
    'cut_frame': 6
}

classTable = {'walk': 0, 'still': 1, 'fall_down': 2, 'stand_up': 3}
model = RNN(input_size=setting['num_features'], hidden_size=setting['hidden_size'],
            num_layers=setting['num_layers'], num_class=len(classTable))
print(model)

model.load_state_dict(torch.load("../python/rnn_weight_lr001stepsize30gamma0.4"))
selected_video = "/home/hanqing/walabot_research/cut_dataset/fall_down/fall_down_0_cut0.avi"
inputs = getFeature2(selected_video, CNN_model, setting['cut_frame'])
inputs = inputs.unsqueeze(1)
output = model(inputs)
output = output.squeeze(1)
_, pred = torch.max(output, 1)
classTensor = torch.Tensor([2])
# classTensor = mapClassToTensor(classTable, classname)
classTensor = classTensor.to(device)

print("output is: \n{}\n"
      "pred is: \n{}\n"
      "class is: \n{}\n".format(output, pred, classTensor.item()))

