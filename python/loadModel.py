import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from python.CNNclassfier import CNN
import cv2
import matplotlib.pyplot as plt
import os


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated

def constructDataSet():
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    # /home/hanqing/walabot_Research/walabotResearch/python
    data_dir = os.path.join(os.path.dirname(cur_dir), 'training_backup/Classfier-position/Test')
    # /home/hanqing/walabot_Research/walabotResearch/training_backup/Classfier/Test

    image_dataset = datasets.ImageFolder(data_dir, data_transforms)

    dataloaders = torch.utils.data.DataLoader(image_dataset, batch_size=4,
                                              shuffle=True, num_workers=4)
    dataset_sizes = len(image_dataset)
    class_names = image_dataset.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return cur_dir, data_dir, image_dataset, dataloaders, dataset_sizes, class_names, device


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


if __name__=="__main__":
    cur_dir, data_dir, image_dataset, dataloaders, dataset_sizes, class_names, device = constructDataSet()

    device = torch.device("cuda:0")
    ori_model = models.resnet18(pretrained=True)
    model = CNN(ori_model)
    model.load_state_dict(torch.load("../python/classfier-position"))
    model = model.to(device)
    visualize_model(model, 12)