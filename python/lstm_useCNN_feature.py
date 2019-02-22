import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from python.utils import *
from python.cnn_model import *

"""
This script takes in multiple frames and output 1 label

benefit1: change avgpool to maxpool
Max pooling extracts the most important features like edges whereas, average pooling extracts features so smoothly.
For image data, you can see the difference. Although both are used for same reason, 
I think max pooling is better for extracting the extreme features. 
Average pooling sometimes can’t extract good features because it takes all into count and results an average value which may/may not be important for object detection type tasks.

benefit2: remove fc layer, keep cnn result

Time: Feb.5.2019
"""
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
        self.fc = nn.Linear(512, 64)

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
        x = x.view(-1,512)
        x = self.fc(x)
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
        out = out[-1, :, :]
        out = F.log_softmax(out, dim=1)

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
        #print(random_order_list, video_dir)
        # Each Epoch Iterate a whole dataset
        for value in random_order_list:
            print(value)
            [cls, video] = value.split()
            selected_video = os.path.join(video_dir, cls, video)
            inputs = getFeature2(selected_video, CNN_model, setting['cut_frame'])
            inputs = inputs.unsqueeze(1)            # change dim from
                                                    # (num_frame, num_features) => (num_frame, 1, num_features)

            model.zero_grad()
            model.train()

            with torch.set_grad_enabled(True):
                print(inputs.size())
                output = model(inputs)
                print(output.size(), output)
                _, pred = torch.max(output, 1)

                classTensor = torch.Tensor([classTable[cls]])

                # classTensor = mapClassToTensor(classTable, classname)
                classTensor = classTensor.to(device)
                # print("OUTPUT IS : \n\n{}\n\n".format(output))
                # print("OUTPUT SIZE IS: \n\n{}\n\n".format(output.size()))
                # print("ClassTensor IS : \n\n{}\n\n".format(classTensor))
                # print("ClassTensor SIZE IS: \n\n{}\n\n".format(classTensor.size()))

                print("output is: \n{}\n"
                      "pred is: \n{}\n"
                      "class is: \n{}\n".format(output, pred.item(),classTensor.item()))


                if (pred.item() == classTensor.item()):
                    correct_count += 1
                loss = criterion(output, classTensor.long())


                loss.backward()
                optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss/ len(random_order_list)
        epoch_acc = correct_count/ len(random_order_list)
        epoch_loss_list.append(epoch_loss)
        print('Loss: {:.4f}'.format(epoch_loss))
        print('Training Accuracy: {:.2f}%\n\n'.format(epoch_acc*100))
        if epoch_acc > best_acc:
            print("save best ")
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            save_path = '../python/rnn_weight'
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(best_model_wts)
    print(best_acc)
    x = np.arange(len(epoch_loss_list))
    plt.plot(x, epoch_loss_list)
    plt.show()

    return model


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
    model = RNN(input_size=setting['num_features'], hidden_size=setting['hidden_size'],
                num_layers=setting['num_layers'], num_class=len(classTable))
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    trained_model = train_model(model,criterion,optimizer, exp_lr_scheduler, current_dir, data_dir,setting,classTable,50)






