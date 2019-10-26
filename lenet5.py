import torch.nn as nn
import torch.nn.functional as F
import torch

class lenet5(nn.Module):
    """docstring for lenet5"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), 1, 2);
        self.conv2 = nn.Conv2d(32, 64, (5, 5), 1, 2);
        self.pool = nn.MaxPool2d((2, 2), (2, 2));
        self.fc1 = nn.Linear(7*7*64, 1024);
        self.fc2 = nn.Linear(1024, 10);
        self.dropout = nn.Dropout(0.5);

    def forward(self, x):
        x = x.view(-1, 1, 28, 28);
        x = self.pool(F.relu(self.conv1(x)));
        x = self.pool(F.relu(self.conv2(x)));
        x = x.view(-1, 7*7*64);
        x = F.relu(self.fc1(x));
        x = self.dropout(x);
        x = self.fc2(x);
        return x

    def loss_fn(self):
        return nn.CrossEntropyLoss();

    def get_accurancy(self, output, labels):
        result = (output*labels).sum(dim=1)
        total = list(output.shape)[0]

        return result.sum().item()/total
