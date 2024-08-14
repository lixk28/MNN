import torch
import torch.nn as nn
import torch.onnx as onnx

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

net = Net()
net.to(device='cpu')
net.eval()

x = torch.rand(1, 3, 4, 4, device='cpu')
traced_net = torch.jit.trace(net, x)

onnx.export(
    traced_net,
    x,
    "net.onnx",
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    verbose=True,
    opset_version=13,
)
