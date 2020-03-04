# Use argparse to parse commandline arguments
import argparse


parser = argparse.ArgumentParser(description="Train my neural network to classifiy flowers.")

parser.add_argument('data_dir', nargs='?', action="store", default="./flowers/")
parser.add_argument('--save_dir', nargs='?', action="store", default="./model.pth",  dest='save_dir')
parser.add_argument('--arch', nargs='?', action="store", default="vgg16", dest="arch")
parser.add_argument('--learning_rate', nargs='?', action="store", default=0.001, type=int, dest="lr")
parser.add_argument('--hidden_units', nargs='?', action="store", default=500, type=int, dest="hidden_units")
parser.add_argument('--epochs', nargs='?', action="store", default="5", type=int, dest="epochs")
parser.add_argument('--gpu', action="store_true", default=True,  dest="gpu")

# Parse the input arguments to get the namespaces object.
args = parser.parse_args()

print("Start Deep Learning process with the following arguments: ")
print(args)

# Import ML libs
import torch
import numpy as np
import torch.nn.functional as F # The activation funtions
from torchvision import datasets, transforms, models
from torch import nn




def loadDatasetAndLoaders(data_dir, batch_size, exp_mean, exp_std, max_img_size):
    """
    Load the datasets from input folder and then return the loaders.

    Args:
    data_dir (string) : a path to the dir location stores the training, validation, testing images.
    batch_size (int) : The batch size for every iteration.
    exp_mean (list(int)) : Expect mean for image normalization.
    exp_std (list(int)) : Expect standar derivation for image normalization.
    max_img_size (int) : The size for resziing.

    Returns:

    image_datasets: A dict contains all Imagefolder.
    
    dataloaders: A dict contains iterable dataloaders.

    """
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
        "training": transforms.Compose([
            transforms.RandomRotation(25), # Rotate degree
            transforms.RandomResizedCrop(max_img_size), # Resising
            transforms.RandomHorizontalFlip(), # flip horizontally
            transforms.ToTensor(), # Covert into Tensor
            transforms.Normalize(exp_mean, exp_std)
        ]),
        # validation and testing have the same transforms.
        "validation": transforms.Compose([
            transforms.Resize(max_img_size+1),
            transforms.CenterCrop(max_img_size),
            transforms.ToTensor(),
            transforms.Normalize(exp_mean, exp_std)
        ]),
        "testing": transforms.Compose([
            transforms.Resize(max_img_size+1),
            transforms.CenterCrop(max_img_size),
            transforms.ToTensor(),
            transforms.Normalize(exp_mean, exp_std)
        ])
    }
    print("Load data from: " + data_dir)
    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        "training" : datasets.ImageFolder(train_dir, transform=data_transforms["training"]),
        "validation": datasets.ImageFolder(valid_dir, transform=data_transforms["validation"]),
        "testing": datasets.ImageFolder(test_dir, transform=data_transforms["testing"])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "training" : torch.utils.data.DataLoader(image_datasets["training"], batch_size=batch_size, shuffle=True),
        "validation": torch.utils.data.DataLoader(image_datasets["validation"],batch_size=batch_size),
        "testing": torch.utils.data.DataLoader(image_datasets["testing"], batch_size=batch_size)
    }
    
    print("----------- Dataset summaray -----------")
    print("Batch size:", batch_size)
    print('training images size: ', len(image_datasets['training']))
    print('validation images size: ', len(image_datasets['validation']))
    print('testing images size: ', len(image_datasets['testing']))
    
    return image_datasets, dataloaders


def buildModel(arch = 'vgg16', hidden_units=500, lr=0.001, n_output=102):
    """
    Build a deep neural network model based on pretrained models provided by torchvision.models.
    It must be clear before choossing suitable model to be transfered into our own model.
    The VGG16 has been proved very accuracy. It archieved more than 95% accracy. 
    (https://neurohive.io/en/popular-networks/vgg16/)
    
    From part I we can see VGG16 has 7 layers in its classifier pipeline.
    The hidden layer (position 3) has 4096 units. we can reduce the number.
    We can use replace its original layer with new layer by tunning some hyper-parameters.
    
   Args:
   arch (string) : the model's name (or architect) we choosen to use as pretrained base.
   hidden_units (int): The neural units (petron units) in hidden layer.
   lr (float) : The learning rate when perform gredient desent.
   n_output (int): The output layer must has the same number of units as the labels.
   
   Returns:
   model : The model with mdofiied classifiers.
   criterion: The loss function using NULLoss
   optimizer: The optimizer with Adam algotithm.
    """
    
    # Load the specifc model from torchvision.models
    # We only provide vgg and mnasnet_05 as altenatives.
    # More models can be found at: https://pytorch.org/docs/stable/torchvision/models.html
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
    

    # Create our Lossfucntion(criterion) and optimizer.
    # Since this is a image classfication problem, we chosse the best loss function to be CrossEntropyLoss,
    # We use Adam as optimier set learning rate at 0.0001
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr)

    # Frezz model's parameters during the feature detection phrase.
    for param in model.parameters():
        param.requires_grad = False 

    # Input layer size
    n_input = model.classifier[0].in_features
    # Hidden layer size
    n_hidden = [hidden_units, 100]

    from collections import OrderedDict

    my_classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(n_input, n_hidden[0])),
        ('relu1', nn.ReLU()),
        ('dropout1',nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(n_hidden[0], n_hidden[1])),
        ('relu2', nn.ReLU()),
        ('dropout2',nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(n_hidden[1],n_output)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # Replace classifier
    model.classifier = my_classifier
    
    print('----------- model\'s classifier ----------')
    print('arch:', arch)
    print(model.classifier)
    return model, criterion, optimizer

def train_model(model, criterion, optimizer, epochs, training_loader, validation_loader, gup):
    """
    Train the given model with the training set and validation set.
    
    Args:
    model (torchvision.Model) : The model to be trained.
    criterion (nn.Functional) : The loss fucntion.
    optimizer (nn.Optimizer) : A optimizer.
    training_loader (dataloader): The dataloder to iterate trainning dataset.
    validatiaon_loader (dataloader): The validation's dataloader.
    """
    model.train()
    # Use GPU if it is possible    
    if gup==True and torch.cuda.is_available():
        model.cuda()
        print("Successful to Use GPU to train model")
    else:
        model.cpu()
        print("Failed to use GPU, it uses CPU now.")
        
    # Determination between CPU and GUP
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
    for e in range(epochs):
        running_loss = 0
        
        for images, labels in training_loader:
            # Move images and labels to device
            images = images.to(device)
            labels = labels.to(device)
            
            # reset the gredients
            optimizer.zero_grad()
            
            # Forward and backward propagation
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            # running loss
            running_loss += loss.item()
            
        else:
            model.eval()
            accuracy = 0
            vaild_loss  = 0
            with torch.no_grad():
                for images, labels in validation_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    log_ps = model(images)
                    vaild_loss += criterion(log_ps, labels)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
        
            running_loss = running_loss / len(training_loader)
            valid_loss = vaild_loss / len(validation_loader)
            accuracy = accuracy / len(validation_loader)
            print("epoch {0}/{} Training loss: {1} ".format(e+1, epochs, running_loss),
                 "Vaildation loss: {}".format(valid_loss),
                 "Accurancy:{}".format(accuracy))
            
def save_model_state(model, optimizer, epoch, training_dataset, file='model.pth'):
    """
    Save model states on persistant storage by using torch.save.
    
    Args:
    model (nn.model) : The model to be saved.
    optimizer (nn.optimizer): The optimizer used to train the model.
    epoch (in) : Training epoch number.
    training_dataset (datasets): The training dataset must be saved to retrain the model status.
    file (string): The path to store the model state.
    """
    model.class_to_idx = training_dataset.class_to_idx
    model_state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx
    }
    torch.save(model_state, file)
    print("Save model to: " , file)
    

# The constants used
exp_mean = [0.485, 0.456, 0.406]
exp_std = [0.229, 0.224, 0.225]
max_img_size = 224
batch_size = 100

# First load datasets.
image_datasets, dataloaders = loadDatasetAndLoaders(args.data_dir, batch_size, exp_mean, exp_std, max_img_size) 
# Then get the model.
my_model, criterion, optimizer = buildModel(args.arch, args.hidden_units, args.lr)
# Train model.
train_model(my_model, criterion, optimizer, args.epochs, dataloaders["training"], dataloaders["validation"], args.gpu)
# Save the model.
save_model_state(my_model, optimizer, args.epochs, image_datasets["training"], args.save_dir)
