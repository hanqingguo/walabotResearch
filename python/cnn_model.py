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
from python.utils import *


def getFeature(input_video, model_ft, num_feature, cut_frame):
    """

    :param input_video: frames feed to model
    :param model_ft:  cnn_models initialized before
    :param num_feature: num of feature you want to output for 1 frame
    :param cut_frame: num of frame to be cut in original video
    :return: stack of features,
        suppose one frame has output feature (1,100)
        then input_video has 10 frames, then output shape is (10,100)
        use torch.cat to concatenate features
    """
    device = torch.device("cuda:0")
    cap = cv2.VideoCapture(input_video)
    n = 0
    features = torch.rand(1, num_feature)       # will be delete this feature at return
    features = features.to(device)

    while (cap.isOpened()) and n<cut_frame:
        _, frame = cap.read()
        rescale = Rescale((224, 224))
        frame = rescale(frame)
        toTensor = ToTensor()
        frame = toTensor(frame)          #        numpy image: H x W x C
                                         #        torch image: C X H X W
        frame = torch.from_numpy(frame)  # ndarray to tensor
        frame = frame.unsqueeze(0)       # from (3,224,224) => (1,3,224,224)
        frame = frame.to(device)

        feature = model_ft(frame.float())
        # feature.size() = (1, num_feature)
        features = torch.cat((features, feature),0)
        n = n + 1

    return features[1:,:]               # return everything except the very begining random feature
                                        # return size = (cut_frame, num_feature)

def getFeature2(input_video, model_ft, cut_frame):
    """

    This function works for lstm_useCNN_feature,
    The difference is remove num_feature, because the last layer of model_ft is pooling layer,
    rather than customized fc layer, the num_feature will keep to 512

    :param input_video: frames feed to model
    :param model_ft:  CNN object in lstm_useCNN_feature.py
    :param num_feature: num of feature you want to output for 1 frame
    :param cut_frame: num of frame to be cut in original video
    :return: stack of features,
        suppose one frame has output feature (1,100)
        then input_video has 10 frames, then output shape is (10,100)
        use torch.cat to concatenate features
    """
    device = torch.device("cuda:0")
    cap = cv2.VideoCapture(input_video)
    n = 0
    features = torch.rand(1, 512)       # will be delete this feature at return
    features = features.to(device)

    while (cap.isOpened()) and n<cut_frame:
        _, frame = cap.read()
        rescale = Rescale((224, 224))
        frame = rescale(frame)
        toTensor = ToTensor()
        frame = toTensor(frame)          #        numpy image: H x W x C
                                         #        torch image: C X H X W
        frame = torch.from_numpy(frame)  # ndarray to tensor
        frame = frame.unsqueeze(0)       # from (3,224,224) => (1,3,224,224)
        frame = frame.to(device)

        feature = model_ft(frame.float())
        # feature.size() = (1, 512, 1, 1)
        feature = feature.squeeze(2)
        feature = feature.squeeze(2)
        # feature.size() = (1, 512)
        features = torch.cat((features, feature),0)
        n = n + 1

    return features[1:,:]               # return everything except the very begining random feature
                                        # return size = (cut_frame, num_feature)



if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    training_dir = os.path.join(os.path.dirname(current_dir), 'training')
    label = 'walk'
    idx = '1'
    video_format = '.avi'
    video_name = idx+video_format
    video_path = os.path.join(training_dir, label, video_name)
    getFeature(video_path, 'resnet18', 100, 10)




