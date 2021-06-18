import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from util import load_data, train_model, test_model
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', action = 'store', dest = 'data_directory', default = './flowers')
parser.add_argument('--save_dir', action = 'store', dest = 'save_directory', default = 'checkpoint.pth')
parser.add_argument('--arch', action='store', default = 'vgg16')
parser.add_argument('--learning_rate', action = 'store', dest = 'learning_rate', type=int, default = 0.001)
parser.add_argument('--dropout', action = 'store', dest='dropout', type=int, default = 0.05)
parser.add_argument('--hidden_layers', action = 'store', dest = 'hidden_layers', type=int, default = 512)
parser.add_argument('--epochs', action = 'store', dest = 'epochs', type = int, default = 4)
parser.add_argument('--gpu', action = "store_true", default = True)

args = parser.parse_args()
data_dir = args.data_directory
save_dir = args.save_directory
learning_rate = args.learning_rate
dropout = args.dropout
hidden_layers = args.hidden_layers
epochs = args.epochs
gpu_mode = args.gpu
arch = args.arch

# Load and process images
train_loader, valid_loader, test_loader, train_data, test_data, valid_data = load_data(data_dir)


if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_layers)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_layers, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

elif arch == 'alexnet':
    model = models.alexnet(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(9216, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(dropout)),
        ('fc2', nn.Linear(4096, hidden_layers)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(dropout)),
        ('fc3', nn.Linear(hidden_layers, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))


for param in model.parameters():
    param.requires_grad = False

model.classifier = classifier

optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

criterion = nn.NLLLoss()

model, optimizer = train_model(model, epochs, train_loader, valid_loader, criterion, optimizer, gpu_mode)

test_model(model, test_loader, gpu_mode)

model.cpu()
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': arch,
              'class_to_idx': train_data.class_to_idx
             }

torch.save (checkpoint, save_dir)
