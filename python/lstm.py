import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from python.utils import *
from python.cnn_model import *


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
        out = out[-1, :, :]
        out = F.log_softmax(out, dim=1)


        return out

def train_model(model, cnn_type, num_feature, criterion, optimizer, exp_lr_scheduler,current_dir, setting, classTable, num_epochs=1000):
    """

    :param model: lstm model
    :param criterion: loss function
    :param optimizer: optimizer
    :param exp_lr_scheduler: learning rate scheduler
    :param current_dir: os.path.dirname(os.path.realpath(__file__))
                        current_dir = /home/hanqing/walabot_Research/walabotResearch/python
    :param setting: cnn models, lstm settings. Dictionary
    :param num_epochs: num_epochs
    :return:
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000
    correct_count = 0
    loss_list = []
    device = torch.device("cuda:0")
    model_ft = getattr(models, cnn_type)(pretrained=True)
    #print(model_ft.parameters)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_feature)
    model_ft = model_ft.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print("-" * 20)
        #exp_lr_scheduler.step()

        running_loss = 0.0
        classname, selected_video = select_video(current_dir, training=True)
        inputs = getFeature(selected_video, model_ft, setting['num_features'], setting['cut_frame'])
        inputs = inputs.unsqueeze(1)
        # inputs.size() = (num_frame, num_features)
        #print(inputs, inputs.size())
        model.zero_grad()
        #model.train()

        with torch.set_grad_enabled(True):
            output = model(inputs)
            _, pred = torch.max(output, 1)

            classTensor = torch.Tensor([classTable[classname]])
            # classTensor = mapClassToTensor(classTable, classname)
            classTensor = classTensor.to(device)
            # print("OUTPUT IS : \n\n{}\n\n".format(output))
            # print("OUTPUT SIZE IS: \n\n{}\n\n".format(output.size()))
            # print("ClassTensor IS : \n\n{}\n\n".format(classTensor))
            # print("ClassTensor SIZE IS: \n\n{}\n\n".format(classTensor.size()))
            #print(classTensor)
            print("output is: \n{}\n"
                  "pred is: \n{}\n"
                  "class is: \n{}\n".format(output, pred.item(),classTensor.item()))
            if (pred.item() == classTensor.item()):
                correct_count += 1
            loss = criterion(output, classTensor.long())
            # for i,param in enumerate(model.parameters()):
            #     print(i, param, param.size())

            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        epoch_loss = running_loss/ len(classTable)
        loss_list.append(epoch_loss)
        epoch_acc = correct_count/ (epoch+1)
        print('Loss: {:.4f}'.format(epoch_loss))
        print('Training Accuracy: {:.2f}%\n\n'.format(epoch_acc*100))
        if epoch_loss < best_loss:
            print("save best ")
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    print(best_loss)
    model.load_state_dict(best_model_wts)
    x = np.arange(len(loss_list))
    plt.plot(x, loss_list)
    plt.show()

    return model

def mapClassToTensor(classTable, classname):
    """

    :param classTable: dictionary. key is classname, value is class index
    :param classname: classname
    :return: (1, classes) Tensor.
            for example, classname is "walk", total class is 5, then return:
            [[1, 0, 0, 0, 0]]
            classname is "sit-to-stand", then return:
            [[0, 1, 0, 0, 0]]
    """
    index = classTable[classname]
    classTensor = torch.zeros(1, len(classTable))
    classTensor[0, index] = 1

    return classTensor

if __name__ == '__main__':
    device = torch.device("cuda:0")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    #/home/hanqing/walabot_research/python
    video_dir = video_dir = os.path.join(os.path.dirname(current_dir), 'test_dataset')

    setting = {'cnn_model': 'resnet18',
               'sequence_num': 2,
               'hidden_size': 6,
               'num_layers': 1,
               'num_directions': 1,
               'num_features': 1,
               'cut_frame': 2}
    """
    sequence_num: how many frame in the sequence
    hidden_size: hidden_size is hidden state dimension
    num_layers: upper layer of each lstm element
    num_directions: 2 is bidirection, 1 is only one direction
    """
    #classTable = {'walk':0, 'sit-to-stand':1, 'stand-to-sit':2, 'fall_down':3, 'jump':4}
    #classTable = {'walk':0, 'sit-to-stand':1, 'stand-to-sit':2, 'jump':3}
    classTable = {'sit-to-stand': 0, 'walk': 1}

    model = RNN(input_size=setting['num_features'], hidden_size=setting['hidden_size'],
                num_layers=setting['num_layers'], num_class=len(classTable))
    model = model.to(device)

    #select_class, selected_video = select_video(current_dir)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(model, setting['cnn_model'], setting['num_features'], criterion, optimizer, exp_lr_scheduler,current_dir, setting, classTable, num_epochs=1000)

