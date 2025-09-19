import torch
import torch.nn as nn
import torch.nn.functional as F


def get_fc(inp_dim, out_dim, non_linear='relu'):
    """
    Mid-level API. It is useful to customize your own for large code repo.
    :param inp_dim: int, intput dimension
    :param out_dim: int, output dimension
    :param non_linear: str, 'relu', 'softmax'
    :return: list of layers [FC(inp_dim, out_dim), (non linear layer)]
    """
    layers = []
    layers.append(nn.Linear(inp_dim, out_dim))
    if non_linear == 'relu':
        layers.append(nn.ReLU())
    elif non_linear == 'softmax':
        layers.append(nn.Softmax(dim=1))
    elif non_linear == 'none':
        pass
    else:
        raise NotImplementedError
    return layers


class SimpleCNN(nn.Module):
    """
    Model definition
    """
    def __init__(self, num_classes=10, inp_size=64, c_dim=3):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=c_dim, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.nonlinear = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flat_dim = 64 * (inp_size // 4) * (inp_size // 4)

        # Sequential is another way of chaining the layers.
        self.fc1 = nn.Sequential(*get_fc(self.flat_dim, 128, 'none'))
        self.fc2 = nn.Sequential(*get_fc(128, num_classes, 'none'))

    def forward(self, x):
        """
        :param x: input image in shape of (N, C, H, W)
        :return: out: classification logits in shape of (N, Nc)
        """

        N = x.size(0)
        x = self.conv1(x)  # (N, 32, H, W)
        x = self.nonlinear(x)  # (N, 32, H, W)
        x = self.pool1(x)  # (N, 32, H/2, W/2)

        x = self.conv2(x)  # (N, 64, H/2, W/2)
        x = self.nonlinear(x)  # (N, 64, H/2, W/2)
        x = self.pool2(x)  # (N, 64, H/4, W/4)

        flat_x = x.view(N, self.flat_dim)
        out = self.fc1(flat_x)  # (N, 128)
        out = self.fc2(out)  # (N, Nc)
        return out
