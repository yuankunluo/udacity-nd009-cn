import argparse


parser = argparse.ArgumentParser(description="Classify a flower using my neural network")
parser.add_argument('image_path', type=str, action="store")
parser.add_argument('checkpoint', type=str, action="store")
parser.add_argument('arch', nargs='?', default="vgg16", action="store")
parser.add_argument('topk', nargs='?', type=int, default=5, action="store")

args = parser.parse_args()


# Import ML libs
import torch
import numpy as np
import torch.nn.functional as F # The activation funtions
from torchvision import datasets, transforms, models
from torch import nn

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_model_by_state(file, arch):
    """
    Load the stored model from storage.
    
   Args:
   file (string): The path to the checkpoint (saved model status).
   arch (string): The name (archited name) or the pretrained model.
   
   Returns:
   model: the pretrained model with our newly trainning states.
   
    """
    
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch =='mnasanet':
        model = models.mnasnet0_5(pretrained=True)
    else:
        model = model = models.vgg16(pretrained=True)
    
    model_state = torch.load(file, map_location= lambda storage, loc : storage)
    
    model.classifier = model_state['classifier']
    model.load_state_dict(model_state['state_dict'])
    model.class_to_idx = model_state['class_to_idx']
    
    print("Load stored model successfully: ", file)
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    exp_mean = [0.485, 0.456, 0.406]
    exp_std = [0.229, 0.224, 0.225]
    max_img_size = 224
    
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(max_img_size),
        transforms.ToTensor(),
        transforms.Normalize(exp_mean, exp_std)
    ])
    
    
    from PIL import Image
    
    image = Image.open(image)
    image = transform(image)
    return image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    # Determination between CPU and GUP
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # get image as tensor
    img = process_image(image_path)
    img = img.to(device)
    img = img.unsqueeze_(0)
    # covert tensor to flot tensor for computing
    img = img.float()
    with torch.no_grad():
        log_ps = model.forward(img)
    probability = torch.exp(log_ps)
    prob, label = probability.topk(topk)
    classes_index = []
    for i in np.array(label[0]):
        for idx, num in model.class_to_idx.items():
            if num == i:
                classes_index += [idx]
    labels = [cat_to_name[str(i)] for i in classes_index]
    probs = prob.view(-1).tolist()
    
    print("----- Prediction Reulst -------")
    print("Input image:", image_path)
    print("\n{:<10} {}".format("Label", "Probability"))
    print("{:<10} {}".format("------", "-----------"))
    for i in range(0, len(labels)):
        print("{:<10} {:.2f}".format(labels[i], probs[i]))
    
    return probs, labels

# First Load model

my_model = load_model_by_state(args.checkpoint, args.arch)
probs, labels = predict(args.image_path, my_model, args.topk)
