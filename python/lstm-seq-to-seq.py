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
        self.hidden2tag = nn.Linear(hidden_size, num_class)


    def forward(self, inputs):
        h_0 = torch.randn(self.num_layers, 1, self.hidden_size)
        c_0 = torch.randn(self.num_layers, 1, self.hidden_size)
        h_0 = nn.init.orthogonal_(h_0)
        c_0 = nn.init.orthogonal_(c_0)
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        out, _ = self.lstm(inputs,(h_0,c_0))
        out = self.hidden2tag(out)

        out = out.squeeze(1)
        #out = self.classifier(out)
        #out = out[-1, :, :]
        out = F.softmax(out, dim=1)


        return out


def train_model(model, criterion, optimizer, exp_lr_scheduler,current_dir, data_dir, setting, classTable, num_epochs):
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
    epoch_loss_list = []
    device = torch.device("cuda:0")
    ori_model = models.resnet18(pretrained=True)
    CNN_model = CNN(ori_model)
    CNN_model = CNN_model.to(device)
    best_acc = 0

    for epoch in range(num_epochs):
        correct_count = 0
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print("-" * 20)

        running_loss = 0.0
        random_order_list, video_dir = video_loader(current_dir, data_dir)
        for value in random_order_list:
            [cls, video] = value.split()
            video_name = cls + "-" + video[:-4]   # get video name in csv file: eg: jump-1
            selected_video = os.path.join(video_dir, cls, video)
            #print(selected_video)

            target_list = target_encoder(video_name, lines, activity_ix)
            target = mapClassToVector(target_list)
            target = target.to(device)
            inputs = getFeature2(selected_video, CNN_model, setting['cut_frame'])
            inputs = inputs.unsqueeze(1)    # change dim from
                                            # (num_frame, num_features) => (num_frame, 1, num_features)
            model.zero_grad()
            model.train()

            with torch.set_grad_enabled(True):
                output = model(inputs)
                _, pred = torch.max(output, 1)
                print("pred is: \n{}\n"
                      "target is: \n{}\n".format(pred, target_list))
                #print("output is: \n{}\n".format(output))
                #print("Target SIZE IS: \n\n{}\n\n".format(target.size()))   # target.size() = [sequence_num]

                #print("OUTPUT SIZE IS: \n\n{}\n\n".format(output.size()))   # output.size() = [sequence_num, act_class]

                loss = torch.norm(output - target)
                print("Current loss is {:.2f}\n".format(loss))
                loss.backward()
                optimizer.step()




    #     classname, selected_video, random_video_idx = select_video(current_dir, training=True)
    #     video_name = classname + "-" + str(random_video_idx + 1)
    #     inputs = getFeature(selected_video, model_ft, setting['num_features'], setting['cut_frame'])
    #     inputs = inputs.unsqueeze(1)
    #
    #     target = target_encoder(video_name, lines, activity_ix)
    #     target = torch.Tensor(target)
    #     target = target.to(device)
    #     #print(inputs)
    #     model.zero_grad()
    #
    #     with torch.set_grad_enabled(True):
    #         output = model(inputs)
    #         _, pred = torch.max(output, 1)
    #
    #         # classTensor = torch.Tensor([classTable[classname]])
    #         # # classTensor = mapClassToTensor(classTable, classname)
    #         # classTensor = classTensor.to(device)
    #
    #         # print("OUTPUT IS : \n\n{}\n\n".format(output))
    #         # print("Target SIZE IS: \n\n{}\n\n".format(target.size()))
    #         # print("OUTPUT SIZE IS: \n\n{}\n\n".format(output.size()))
    #         # print("ClassTensor IS : \n\n{}\n\n".format(classTensor))
    #         # print("ClassTensor SIZE IS: \n\n{}\n\n".format(classTensor.size()))
    #         #print(classTensor)
    #         print("output is: \n{}\n"
    #               "pred is: \n{}\n"
    #               "target is: \n{}\n".format(output, pred , target))
    #
    #         loss = criterion(output, target.long())
    #         loss.backward()
    #         optimizer.step()
    #     running_loss += loss.item()
    #     epoch_loss = running_loss/ len(activity_ix)
    #     loss_list.append(epoch_loss)
    #     #epoch_acc = correct_count/ (epoch+1)
    #     print('Loss: {:.4f}'.format(epoch_loss))
    #     #print('Training Accuracy: {:.2f}%\n\n'.format(epoch_acc*100))
    #     if epoch_loss < best_loss:
    #         print("save best ")
    #         best_loss = epoch_loss
    #         best_model_wts = copy.deepcopy(model.state_dict())
    # print(best_loss)
    # model.load_state_dict(best_model_wts)
    # x = np.arange(len(loss_list))
    # plt.plot(x, loss_list)
    # plt.show()

    return model

def mapClassToTensor(classTable, classname):
    """

    :param classTable: dictionary. key is classname, value is class index
    :param classname: classname
    :return: (1, classes) Tensor.
            for example, classname is "walk", value is 0, total classes is 5, then return:
            [[1, 0, 0, 0, 0]]
            classname is "sit-to-stand", then return:
            [[0, 1, 0, 0, 0]]
    """
    index = classTable[classname]
    classTensor = torch.zeros(1, len(classTable))
    classTensor[0, index] = 1

    return classTensor

def mapClassToVector(target_list):
    """

    :param target_list: a sequece of activity [0, 1, 2, 0 ,3, 3,...,1]
    :return: convert each value of the list to vector.
        for example: Suppose there are 4 kinds of activities
            0 -> [1, 0, 0, 0]
            1 -> [0, 1, 0, 0]
            3 -> [0, 0, 0, 1]
    """
    sequece_cls = torch.zeros(1, len(activity_ix))
    for cls in target_list:
        cls_at_this_moment = torch.zeros(1, len(activity_ix))
        cls_at_this_moment[0][cls] = 1
        sequece_cls = torch.cat((sequece_cls, cls_at_this_moment))
    return sequece_cls[1:,:]


def select_video(current_dir, training=True):
    """

    :param current_dir: the path upper training
    :return: class name of the video, and the real path of the video
    """
    if(training):
        video_dir = os.path.join(os.path.dirname(current_dir), 'training_backup/training')
    else:
        video_dir = os.path.join(os.path.dirname(current_dir), 'testing')
    classes = os.listdir(video_dir)
    random_class = np.random.randint(len(classes)) # get random class folder
    classname = classes[random_class]
    video_list = os.listdir(os.path.join(video_dir,classname))
    random_video_idx = np.random.randint(len(video_list))
    selected_video = os.path.join(video_dir, classname, video_list[random_video_idx])


    return classname, selected_video, random_video_idx

def activity_to_ix():
    """
    Generate activities dictionary like this:
    {'still': 0, 'jump': 1, 'sitting': 2, 'moving': 3}
    :return: dictionary
    """
    activity_ix = {}
    with open(os.path.join(os.path.dirname(current_dir), 'training_backup/label.csv')) as f:
        head = f.readline()
        lines = f.readlines()
        for line in lines:
            line_list = line[:-1].split(",")
            activities = line_list[1:]
            #print(activities)
            for activity in activities:
                if activity not in activity_ix:
                    #print("add new activity {}".format(activity))
                    activity_ix[activity] = len(activity_ix)
    return activity_ix, head, lines

def target_encoder(video_name, lines, activity_ix):
    """
    :param video_name:
    :return: list of encode activity number
    """
    target = []
    for line in lines:
        line = line[:-1].split(",")
        if (line[0] == video_name):
            for cell in line[1:]:
               target.append(activity_ix[cell])
    return target




if __name__ == '__main__':
    device = torch.device("cuda:0")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    #current_dir = / home / hanqing / walabot_Research / walabotResearch / python
    data_dir = 'training_backup/training'


    setting = {'cnn_model': 'resnet18',
               'sequence_num': 15,
               'hidden_size': 6,
               'num_layers': 1,
               'num_directions': 1,
               'num_features': 512,
               'cut_frame': 15}
    """
    sequence_num: how many frame in the sequence
    hidden_size: hidden_size is hidden state dimension
    num_layers: upper layer of each lstm element
    num_directions: 2 is bidirection, 1 is only one direction
    """
    #classTable = {'walk':0, 'sit-to-stand':1, 'stand-to-sit':2, 'fall_down':3, 'jump':4}
    #classTable = {'walk':0, 'sit-to-stand':1, 'stand-to-sit':2, 'jump':3}
    classTable = {'stand-to-sit': 0, 'jump': 1}
    activity_ix, head, lines = activity_to_ix()
    # activity_ix = {'still': 0, 'jump': 1, 'sitting': 2, 'moving': 3}

    model = RNN(input_size=setting['num_features'], hidden_size=setting['hidden_size'],
                num_layers=setting['num_layers'], num_class=len(activity_ix))
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer, exp_lr_scheduler,current_dir, data_dir, setting, classTable, num_epochs=100)

