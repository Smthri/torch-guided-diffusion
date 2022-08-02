import torch
import torchvision.models as models


class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.fcn = torch.nn.Linear(1000, 9)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fcn(x)
        return x

