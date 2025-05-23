import torchvision.models as models
import torch.nn as nn

class ResNet101(nn.Module):
    def __init__(self, num_classes=31):
        super(ResNet101, self).__init__()
        self.base = models.resnet101(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes),
            nn.ReLU(),
            nn.Softmax()
        )
    
    def forward(self, x):
        features = self.base(x)
        return self.classifier(features)
    
    def idd_loss(self):
        # Theorem 4: Gradient alignment loss
        return torch.nn.L1Loss(reduction='mean')(self.grad1, self.grad2)