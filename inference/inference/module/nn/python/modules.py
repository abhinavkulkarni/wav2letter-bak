from collections import defaultdict

import torch
from torch import nn


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        x = x.view(self.shape)
        return x

    def __repr__(self):
        return "Reshape(" + str(self.shape) + ")"


class Permute(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        x = x.permute(self.permutation)
        return x

    def __repr__(self):
        return "Permute(" + str(self.permutation) + ")"


class W2LGroupNorm(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        dim = 1
        mean = x.mean(dim, keepdims=True)
        std = x.std(dim, unbiased=False, keepdims=True)
        x = (x - mean) / std * self.alpha + self.beta
        return x

    def __repr__(self):
        return f"GroupNorm(alpha={self.alpha}, beta={self.beta})"


class Conv1dUnequalPadding(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, left_padding, right_padding, groups):
        super().__init__(in_channels // groups, out_channels // groups, kernel_size, stride, padding=0, groups=1)
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.left_padding_tensor = torch.zeros((1, in_channels, left_padding)).float()
        self.right_padding_tensor = torch.zeros((1, in_channels, 0)).float()
        self.num_groups = groups

    def forward(self, x):
        x = torch.cat((self.left_padding_tensor, x, self.right_padding_tensor), -1)
        kernel_size = self.kernel_size[0]
        stride = self.stride[0]
        num_out_frames = (x.shape[-1] - kernel_size) // stride + 1
        num_consumed_frames = num_out_frames * stride
        self.left_padding_tensor = x[..., num_consumed_frames:]
        y = torch.empty((1, 0, num_out_frames))
        for i in range(self.num_groups):
            _x = x[:, (i * self.in_channels):((i + 1) * self.in_channels), :]
            _x = super().forward(_x)
            y = torch.cat((y, _x), 1)
        return y

    def finish(self):
        self.right_padding_tensor = torch.zeros((1, self.in_channels * self.num_groups, self.right_padding)).float()

    def __repr__(self):
        s = super().__repr__()
        s = f"{s}\b, padding={self.left_padding, self.right_padding})"
        return s


class Residual(nn.Module):
    def __init__(self, name, module):
        super().__init__()
        self.name = name
        self.module = module
        self.padding = torch.zeros(0).float()

    def forward(self, x):
        _x = self.module.forward(x)

        dim = -1 if len(x.shape) == 3 else 0
        x = torch.cat((self.padding, x), dim)
        size = min(x.shape[dim], _x.shape[dim])
        _x = x.narrow(dim, 0, size) + _x.narrow(dim, 0, size)
        self.padding = x.narrow(dim, size, x.shape[dim] - size)
        return _x


def get_module(obj):
    name = obj['name']
    if name == 'Linear':
        in_features, out_features = obj['inFeatures'], obj['outFeatures']
        return nn.Linear(in_features, out_features)
    elif name == 'Conv1d':
        in_channels, out_channels, kernel_size, stride, left_padding, right_padding, groups = obj['inChannels'], obj[
            'outChannels'], obj['kernelSize'], obj['stride'], obj['leftPadding'], obj['rightPadding'], obj['groups']
        return Conv1dUnequalPadding(in_channels, out_channels, kernel_size, stride, left_padding, right_padding, groups)
    elif name == 'ReLU':
        return nn.ReLU()
    elif name == 'Identity':
        return nn.Identity()
    elif name == 'Sequential':
        counter = defaultdict(int)
        sequential = nn.Sequential()
        for child in obj['children']:
            name = child['name']
            _name = name + '-' + str(counter[name])
            counter[name] += 1
            sequential.add_module(name=_name, module=get_module(child))
        return sequential
    elif name == 'Residual':
        module = get_module(obj['module'])
        return Residual(name, module)
    elif name.startswith('GroupNorm'):
        alpha, beta = obj['alpha'], obj['beta']
        return W2LGroupNorm(alpha, beta)
    elif name == 'Reshape':
        shape = obj['shape']
        return Reshape(shape)
    elif name == 'Permute':
        permutation = obj['permutation']
        return Permute(permutation)
    else:
        raise ValueError("Unknown module type: " + name)
