#******************************************************
# This Script reconstruct resnet18 maxpool layer and
# Linear layer, feed three categories pictures with
# labels {middle, left, right} as position of radar detect
# object. or {ambiguous, regular}
#
#
# middle, left, right
# Or
# regular, ambiguous
#
# ambiguous images vs regular images
# dataset_dir =  /home/hanqing/walabot_Research/walabotResearch/training_backup/Classfier/Train
#
# position classification
# dataset_dir = /home/hanqing/walabot_Research/walabotResearch/training_backup/Classfier-position/Train
#******************************************************






import torch.nn.functional as F
import torch.optim as optim
from python.utils import *
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from python.utils import *


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

        # Reuse original model layer1, layer2, layer3, layer4
        # Reconstruct MaxPool2d after layer4
        # Add Linear 512*3 as 3 classes
        #self.maxpool1 = nn.MaxPool2d(kernel_size=7, stride=1, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc = nn.Linear(512, 2, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.maxpool1(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def constructDataSet():
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    # /home/hanqing/walabot_Research/walabotResearch/python
    data_dir = os.path.join(os.path.dirname(cur_dir), 'training_backup/Classfier/Train')
    # /home/hanqing/walabot_Research/walabotResearch/training_backup/Classfier/Train

    image_dataset = datasets.ImageFolder(data_dir, data_transforms)

    dataloaders = torch.utils.data.DataLoader(image_dataset, batch_size=4,
                                              shuffle=True, num_workers=4)
    dataset_sizes = len(image_dataset)
    class_names = image_dataset.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return cur_dir, data_dir, image_dataset, dataloaders, dataset_sizes, class_names, device

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2)  # pause a bit so that plots are updated

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    path_to_train_loss = "../python/loss_data/train_loss_avgpool.txt"
    path_to_acc = "../python/loss_data/acc_avgpool.txt"

    iteration_loss_list = []
    iteration_acc_list = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)
        scheduler.step()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        current_batch = 0
        for inputs, labels in dataloaders:
            current_batch += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)

                _, preds = torch.max(outputs, 1)

                print("output is \n\n{}\n\n"
                      "predict is \n\n{}\n\n"
                      "label is \n\n{}\n\n".format(outputs, preds, labels))

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            iteration_loss_list.append(running_loss  / current_batch )

            running_corrects += torch.sum(preds == labels.data)
            iteration_acc_list.append(running_corrects.item() / (inputs.size(0)*current_batch))



        epoch_loss = running_loss / dataset_sizes
        epoch_acc = running_corrects.double() / dataset_sizes
        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            save_path = '../python/classfier'
            torch.save(model.state_dict(), save_path)
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    print(iteration_loss_list)
    print(iteration_acc_list)
    with open(path_to_train_loss, 'w') as f:
        for l in iteration_loss_list:
            f.write(str(l)+'\n')
        f.close()
    with open(path_to_acc, 'w') as f:
        for l in iteration_acc_list:
            f.write(str(l)+'\n')
        f.close()

    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

if __name__ == "__main__":

    cur_dir, data_dir, image_dataset, dataloaders, dataset_sizes, class_names, device = constructDataSet()
    ori_model = models.resnet18(pretrained=True)
    model_ft = CNN(ori_model)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=50)
    visualize_model(model_ft,num_images=6)