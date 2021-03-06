import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _make_layers(cfg):
    layers = []
    in_channels = 3
    for layer_cfg in cfg:
        if layer_cfg == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=layer_cfg,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True))
            layers.append(nn.BatchNorm2d(num_features=layer_cfg))
            layers.append(nn.ReLU(inplace=True))
            in_channels = layer_cfg
    return nn.Sequential(*layers)

class _VGG(nn.Module):
    """
    VGG module for 3x32x32 input, 10 classes
    """

    def __init__(self):
        super(_VGG, self).__init__()
        cfg = _cfg['VGG16']
        self.layers = _make_layers(cfg)
        flatten_features = 512
        self.fc1 = nn.Linear(flatten_features, 10)
        # self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        y = self.layers(x)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        # y = self.fc2(y)
        # y = self.fc3(y)
        return y


def load_model():
    net = _VGG()
    net.load_state_dict(torch.load("result/vgg16.pt"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    return net

with open("train_acc.txt") as f:
    train_acc = f.readlines()
with open("test_acc.txt") as f:
    test_acc = f.readlines()
with open("train_loss.txt") as f:
    loss = f.readlines()
#
# you may also want to remove whitespace characters like `\n` at the end of each line
train_acc_list = [float(x.strip()) for x in train_acc] 
test_acc_list = [float(x.strip()) for x in test_acc] 
loss_list = [float(x.strip()) for x in loss] 
x = range(20)
plt.plot(x, loss_list, label='loss')
# plt.plot(x, test_acc_list, label='test_acc')
# plt.plot(x, y2, label='cos')
# plt.text(0.08, 0.2, 'sin')
# plt.text(0.9, 0.2, 'cos')
plt.legend()
plt.show()

