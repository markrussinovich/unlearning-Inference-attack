import requests
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
from sklearn.neighbors import KernelDensity


import torch
from torch import nn
from torch import optim
from torch import utils
from torch.nn import functional as F    
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18

import matplotlib.pyplot as plt
import seaborn as sns

def set_device():
    global DEVICE
    if 'DEVICE' not in globals():
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print("Running on device:", DEVICE.upper())

set_device()

def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


def get_outputs(net, loader, label = None):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    all_outputs = None
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        if label is not None:
            inputs = inputs[targets == label]
        outputs = net(inputs)        
        if all_outputs is None:
            all_outputs = outputs    
        else:
            all_outputs = torch.cat((all_outputs, outputs))

    return all_outputs

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def scatter_class(net, loader, label, title):
    with torch.no_grad():
        outputs = get_outputs(net, loader, label)

    # number of logits, assuming output is logits and shape is (num_samples, num_logits)
    num_logits = outputs.shape[1]  

    # Prepare x and y arrays
    x = []
    y = []
    
    # iterate through each logit
    for i in range(num_logits):
        x += [i]*outputs.shape[0]  # Create a list with the logit index repeated for all samples
        y += outputs[:, i].cpu().numpy().tolist()  # Select the output values for the current logit and convert to list

    # Convert lists to numpy arrays
    x = np.array(x)
    y = np.array(y)
        
    # Create hexbin plot
    plt.hexbin(x, y, gridsize=50, cmap='rainbow', norm=LogNorm())
    plt.colorbar(label='Count')

    plt.xlabel('Logit index')
    plt.ylabel('Output value')
    plt.title(title + ' logit distribution for class ' + str(label))
    plt.show()

def get_unique_classes(loader):
    target_list = []
    for _, targets in loader:
        target_list.extend(targets.tolist())
    return list(set(target_list))

# update the targets to match the current model predictions
def update_retain_targets(net, loader):

    device = next(net.parameters()).device  # Get the device of the model
    new_targets = []
    inputs_list = []
    
    net.eval()  # Set the network in evaluation mode
    with torch.no_grad():  # No need to track gradients in evaluation mode
        for inputs, _ in loader:
            inputs = inputs.to(device)  # Move inputs to the same device as the model
            outputs = net(inputs)
            new_targets.append(outputs.cpu().clone().float())
            inputs_list.append(inputs.cpu().clone().float())

    new_targets = torch.cat(new_targets, dim=0)
    inputs_list = torch.cat(inputs_list, dim=0)
   
    # Create a new TensorDataset and DataLoader with the updated targets
    updated_dataset = TensorDataset(inputs_list, new_targets)
   
    return updated_dataset

def get_test_distribution(net, forget_classes, loader):
    net.eval()  # Set the network in evaluation mode
    device = next(net.parameters()).device  # Get the device of the model
    distribution = {cls: {'logits': [], 'kde': None} for cls in forget_classes}  # Initialize distribution dictionary

    with torch.no_grad():  # No need to track gradients in evaluation mode
        for data, target in loader:
            data = data.to(device)  # Move data to the same device as the model
            output = net(data)  # Forward pass

            for i, cls in enumerate(target):
                if cls.item() in forget_classes:
                    distribution[cls.item()]['logits'].append(output[i].cpu().numpy())

    # Compute KDE for each class
    for cls in distribution:
        logits = np.array(distribution[cls]['logits'])
        distribution[cls]['kde'] = KernelDensity(kernel='gaussian').fit(logits)

    return distribution

def generate_sample(distribution, cls):
    kde = distribution[cls]['kde']
    logit_sample = kde.sample()[0]
    return logit_sample

def update_forget_targets(net, loader, distribution):
    """Create a new DataLoader with targets sampled from the provided distribution."""
    device = next(net.parameters()).device  # Get the device of the model
    new_targets = []
    inputs_list = []
    
    net.eval()  # Set the network in evaluation mode
    with torch.no_grad():  # No need to track gradients in evaluation mode
        for inputs, targets in loader:
            inputs = inputs.to(device)  # Move inputs to the same device as the model
            sampled_targets = []
            for target in targets:
                if target.item() in distribution:
                    sampled_target = generate_sample(distribution, target.item())
                    sampled_targets.append(sampled_target)
            new_targets.append(torch.tensor(np.array(sampled_targets)).float()) 
            inputs_list.append(inputs.cpu().clone())

    new_targets = torch.cat(new_targets, dim=0)
    inputs_list = torch.cat(inputs_list, dim=0)
   
    # Create a new TensorDataset and DataLoader with the sampled targets
    updated_dataset = TensorDataset(inputs_list, new_targets)
    
    return updated_dataset


class TaggedDataset(Dataset):
    """A dataset wrapper that adds a tag to each example."""
    
    def __init__(self, base_dataset, tags):
        assert len(base_dataset) == len(tags), "base_dataset and tags must be the same length"
        self.base_dataset = base_dataset
        self.tags = tags
    
    def __getitem__(self, index):
        x, y = self.base_dataset[index]
        tag = self.tags[index]
        return x, y, tag
    
    def __len__(self):
        return len(self.base_dataset)
    
def finetune_model(net, train_loader):
    loss_progress = []
    epochs = 100
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    net.train()

    batch = 0
    for epoch in range(epochs):
        for inputs, targets, tags in train_loader:
            inputs, targets, tags = inputs.to(DEVICE), targets.to(DEVICE), tags.to(DEVICE)
            outputs = net(inputs)
            optimizer.zero_grad()

            # Apply MSE loss to samples from the retain dataset
            retain_mask = (tags == 0)
            combined_loss = torch.tensor(0.0).to(DEVICE)
            if retain_mask.any():
                #combined_loss = criterion_ce(outputs[retain_mask], targets[retain_mask].argmax(dim=1))
                combined_loss += criterion_mse(outputs[retain_mask], targets[retain_mask])

            # Apply CrossEntropy loss to samples from the forget dataset
            forget_mask = (tags == 1)
            if forget_mask.any():
                # Assuming targets are class indices
                combined_loss += criterion_mse(outputs[forget_mask], targets[forget_mask])

            combined_loss.backward()
            optimizer.step()
            loss_progress.append(combined_loss.item())

            batch += 1
            if batch % 100 == 0:
                print(f"[{epoch}:{batch}] loss: {combined_loss.item()}")

            # Are we done?
            if combined_loss.item() < 0.1:
                print(f"[{epoch}:{batch}] reached acceptable loss: {combined_loss.item()}")
                break                
        scheduler.step()

    net.eval()
    return net, loss_progress

def unlearning(net, retain, forget, validation):
    # get classes in forget set
    print( "Getting unique classes in forget set")
    forget_classes = get_unique_classes(forget)

    # get distributions for test set
    print( "Getting test distribution")
    test_distribution = get_test_distribution(net, forget_classes, validation)

    # update forget set with new targets that match the distribution of the test set
    print( "Updating forget set")
    test_forget = update_forget_targets(net, forget, test_distribution)

    # get an updated version of the retain set that has outputs from the current model
    # we do that to keep the model's performance on the training data stable
    print( "Updating retain set")
    retain_stable = update_retain_targets(net, retain)

    # concatenate retain and forget sets
    retain_dataset_tagged = TaggedDataset(retain_stable, torch.tensor([0] * len(retain_stable)))
    forget_dataset_tagged = TaggedDataset(test_forget, torch.tensor([1] * len(test_forget)))
    train_set = torch.utils.data.ConcatDataset([retain_dataset_tagged, forget_dataset_tagged])

    # shuffle the data
    print( "Shuffling the data")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=512, shuffle=True, num_workers=0
    )

    # train the model
    print( "Training the model")
    ft,loss_progress = finetune_model( net, train_loader )
    print("Done training")
    return ft
    