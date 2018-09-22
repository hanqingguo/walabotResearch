from __future__ import print_function, division

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
import cv2
from utils import *

def train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epoch=25):
    pass


def getFeature(input_video, model_type, num_feature, cut_frame):
    """

    :param input_video: frames feed to model
    :param model_type:  transfer learning model type, for example: resnet18,vgg16 and so on
    :param num_feature: num of feature you want to output
    :param cut_frame: num of frame to be cut in original video
    :return: list of features of each frame. Each feature.shape is (num_feature, 1)
        [features at frame[0], features at frame[1], ... , features at frame[n]]
    """
    device = torch.device("cuda:0")
    model_ft = getattr(models, model_type)(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_feature)
    model_ft = model_ft.to(device)
    cap = cv2.VideoCapture(input_video)
    n = 0
    feature_list = []
    while (cap.isOpened()) and n<cut_frame:
        _, frame = cap.read()
        print(frame, type(frame), frame.shape)
        cv2.imshow('frame',frame)
        cv2.waitKey(1000)
        rescale = Rescale((224, 224))
        frame = rescale(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(1000)
        print(frame.shape)
        exit(0)

        feature = model_ft(frame)
        feature_list.append(feature)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    training_dir = os.path.join(os.path.dirname(current_dir), 'training')
    label = 'walk'
    idx = '1'
    video_format = '.avi'
    video_name = idx+video_format
    video_path = os.path.join(training_dir, label, video_name)
    getFeature(video_path, 'resnet18', 100, 30)




# features = getFeature('resnet18',10)
# print(features)
# device = torch.device("cuda:0")
# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 100)
# model_ft = model_ft.to(device)
#
# criterion = nn.CrossEntropyLoss()
#
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum= 0.9)
#
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25)





