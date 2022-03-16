import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(stride=2, kernel_size=3, padding=1)

        self.layer2 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool2d(stride=2, kernel_size=3, padding=1)

        self.layer3 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.pool3 = nn.AvgPool2d(stride=2, kernel_size=3, padding=1)
        
        self.gru_layer = nn.GRU(input_size=256, hidden_size=256, num_layers=3, batch_first = True)

        self.layer5 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.layer7 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)

    def forward(self, x):
        x = x / 1023.0

        x = self.layer1(x)
        x = torch.relu(self.pool1(x))

        x = self.layer2(x)
        x = torch.relu(self.pool2(x))

        x = self.layer3(x)
        x = torch.relu(self.pool3(x))

        x, h_n = self.gru_layer(x.view(-1, 96, 16*16))
    
        x = torch.relu(self.layer5(x.view(-1, 96, 16, 16)))
        x = torch.sigmoid(self.layer7(x)) * 1023.0

        return x