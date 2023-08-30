
import torch.nn as nn
import torch


# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 搭建网络模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=0),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=0),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)

        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



