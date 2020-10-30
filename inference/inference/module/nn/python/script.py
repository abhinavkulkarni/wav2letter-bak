import argparse
import json
import sys

import torch

from inference.inference.module.nn.python.modules import get_module

parser = argparse.ArgumentParser()
parser.add_argument('--arch-json', help='Model architecture file')
parser.add_argument('--arch-weights', help='LibTorch serialized weights')

args = parser.parse_args(sys.argv[1:])

# Create the model from model architecture file and copy model parameters
with open(args.arch_json) as f:
    obj = json.loads(f.read())

sequential = get_module(obj)
params1 = sequential.parameters()

model = torch.jit.load(args.arch_weights)
params2 = model.parameters()

for p1, p2 in zip(params1, params2):
    p1.data.copy_(p2.data)

# Run sample data
x = torch.arange(57 * 80).float()
x = (x - x.mean()) / x.std()

x = sequential.forward(x).contiguous()
x = x.reshape((-1, 1))
with open("/tmp/output-python.txt", 'w') as f:
    x = x.detach().numpy()
    x = x.squeeze()
    for item in x:
        f.write(f"{item: 2.4e}")
        f.write("\n")

# Run sample data
x = torch.arange(57 * 80).float()
x = (x - x.mean()) / x.std()
scripted_module = torch.jit.script(model.half())
scripted_module.save('/tmp/acoustic_model_half_scripted.pth')
