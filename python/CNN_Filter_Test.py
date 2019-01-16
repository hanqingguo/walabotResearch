from python.CNNclassfier import CNN
from python.CNNclassfier import *
from python.CNN_Filter import *

cur_dir, data_dir, image_dataset, dataloaders, dataset_sizes, class_names, device = constructDataSet()
ori_model = models.resnet18(pretrained=True)
model_ft = CNN(ori_model)
model_ft = model_ft.to(device)

model_ft.load_state_dict(torch.load("../python/classfier"))

for (inputs, _) in dataloaders:
    inputs = inputs.to(device)
    result = filter(model_ft, inputs)
