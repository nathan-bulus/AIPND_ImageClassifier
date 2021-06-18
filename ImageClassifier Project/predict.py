import argparse
import torch
import numpy as np
from torchvision import datasets, transforms, models
import json
from PIL import Image

parser = argparse.ArgumentParser(description='Take the path to an image and a checkpoint, then return the top K most probably classes for that image')

parser.add_argument('--image_dir', action='store', default='./flowers/test/24/image_06849.jpg', type=str)
parser.add_argument('--save_dir', action='store', default='checkpoint.pth', type=str)
parser.add_argument('--topk', action='store', dest='topk', type=int, default=5)
parser.add_argument('--cat_to_name', action='store', dest='cat_to_name', default='cat_to_name.json')
parser.add_argument('--gpu', action="store_true", default=True)

args = parser.parse_args()

image_dir = args.image_dir
save_dir = args.save_dir
topk = args.topk
cat_name = args.cat_to_name
gpu_mode = args.gpu



checkpoint = torch.load(save_dir)
if checkpoint['arch'] == 'vgg16':
    model = models.vgg16(pretrained=True)

elif checkpoint['arch'] == 'alexnet':
    model = models.alexnet(pretrained=True)

model.state_dict (checkpoint['state_dict'])
model.classifier = checkpoint['classifier']
model.class_to_idx = checkpoint['class_to_idx']

for param in model.parameters():
    param.requires_grad = False

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def process_image(image_dir):
    im = Image.open(image_dir)
    im = im.resize((256,256))
    crop_value = 0.5*(256-224)
    im = im.crop((crop_value, crop_value, 256-crop_value, 256-crop_value))

    np_image = np.array(im)/255

    im_mean = [0.485, 0.456, 0.406]
    im_std = [0.229, 0.224, 0.225]

    np_image = (np_image - im_mean) / im_std

    np_image = np_image.transpose(2, 0, 1)

    return np_image


def predict(image_dir, model, topk, gpu_mode):
    torch_image = process_image(image_dir)

    if gpu_mode == True:
        model.to('cuda')
    else:
        model.cpu()



    torch_image = torch.from_numpy(torch_image).type(torch.FloatTensor)
    torch_image = torch_image.unsqueeze(0)
    torch_image = torch_image.float()
    if gpu_mode == True:
        torch_image = torch_image.to('cuda')
    else:
        pass

    output = model.forward(torch_image)
    linear_probs = torch.exp(output)

    #Find the top 5 results
    top_probs, top_labels = linear_probs.topk(topk)

    # Detach all of the details
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]

    idx_to_class = {val: key for key, val in
                   model.class_to_idx.items()}
    top_labels = [idx_to_class[label] for label in top_labels]
    top_flowers = [cat_to_name[label] for label in top_labels]

    return top_probs, top_labels, top_flowers


probs, classes, name = predict(image_dir, model, topk, gpu_mode)

names = [cat_to_name[i] for i in classes]

print(f"'{names[0]}' with a probability of {round(probs[0]*100,2)}% ")
