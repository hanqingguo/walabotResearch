import torchvision.models as models
import torch.nn as nn

"""
试试LRCN同样的配置能不能学到东西

Time: Feb.22.2019

http://wulc.me/2018/04/18/%E9%80%9A%E8%BF%87%20Keras%20%E5%AE%9E%E7%8E%B0%20LRCN%20%E6%A8%A1%E5%9E%8B/
"""

vgg16 = models.vgg16(pretrained=True)
print(vgg16.training)


class LRCN_model(nn.Module):
    def __init__(self):
        super(LRCN_model, self).__init__()
        self.cnn = vgg16
        self.fc1 = nn.Linear()

