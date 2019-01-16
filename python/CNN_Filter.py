################################################################
# This script define a filter function, inputs is current frame,
# passing CNN in CNNclassfier.py and returns true or false.
# True means input is regular frame
# False means inputs is ambiguous frame
################################################################
import torch
from python.CNNclassfier import *

def filter(model, inputs):
    outputs = model(inputs)
    #imshow(inputs.cpu().data[0])
    _, outputs = torch.max(outputs, 1)
    output = outputs[0]
    result = outputMap(output)
    return result

def outputMap(output):
    if output.item() == 0:
        print("bad")
        return False
    else:
        print("regular")
        return True