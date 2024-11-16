import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTNet
from datetime import datetime
import os
import torch.nn.functional as F

SEED = 1

def train(is_aug=False):
    # Set device to CPU
    device = torch.device("cpu")

    # For reproducibility
    torch.manual_seed(SEED)
    
    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if is_aug:
        transform = transforms.Compose([
                                       transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                       transforms.RandomAffine(degrees=0,  scale=(0.95, 1.05)), 
                                       transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                                       ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    
    model = MNISTNet().to(device)
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.15, momentum=0.9) 
    
    # Training
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) #criterion(output, target)
        loss.backward()
        optimizer.step()
        
        predicted =  output.argmax(dim=1, keepdim=True) #output.max(1)
        total += target.size(0)
        correct += predicted.eq(target.view_as(predicted)).sum().item()
        
    accuracy = 100. * correct / total
    print(f'Training Accuracy: {accuracy:.2f}%')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'model_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    return accuracy, model

if __name__ == "__main__":
    train() 