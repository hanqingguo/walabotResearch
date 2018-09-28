import torch
import torch.nn as nn
from torchvision import models
import cv2
from python.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


# 预测值f(x) 构造样本，神经网络输出层
inputs_tensor = torch.FloatTensor( [
[-0.0678, -0.4793,  0.1375, -0.1964]
 ])

# 真值y
targets_tensor = torch.LongTensor([2])
# targets_tensor = torch.LongTensor([1])

inputs_variable = autograd.Variable(inputs_tensor, requires_grad=True)
targets_variable = autograd.Variable(targets_tensor)
print('input tensor(nBatch x nClasses): {}'.format(inputs_tensor.shape))
print('target tensor shape: {}'.format(targets_tensor.shape))

loss = nn.CrossEntropyLoss()
output = loss(inputs_variable, targets_variable)
# output.backward()
print('pytorch 内部实现的CrossEntropyLoss: {}'.format(output))