import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride, padding):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.body(x)
        return x
    
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNAct(3, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBNAct(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # test LeNet
    net = LeNet()
    dummy_input = torch.randn(16, 3, 32, 32)
    dummy_output = net(dummy_input)
    print(f"# intput shape = {dummy_input.shape}")
    print(f"# output shape = {dummy_output.shape}")