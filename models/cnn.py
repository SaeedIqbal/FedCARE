import torch.nn as nn

class MSKCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MSKCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8192, 3072), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(3072, 100), nn.ReLU(),
            nn.Linear(100, num_classes), nn.Softmax()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features.view(features.shape[0], -1))