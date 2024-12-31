#! /usr/bin/env python
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets
import os
from timm.loss import LabelSmoothingCrossEntropy

dataset_path = './data'
classes = datasets.ImageFolder(os.path.join(dataset_path, 'train')).classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HUB_URL = "SharanSMenon/swin-transformer-hub"
MODEL_NAME = "swin_tiny_patch4_window7_224"
# check hubconf for more models.
model: nn.Module = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True) # load from torch hub
n_input = model.head.in_features

FINTUNE = True
if FINTUNE:
    for param in model.parameters():
        param.requires_grad = False
    
model.head = nn.Sequential(
    nn.Linear(n_input, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(classes))
)
model = model.to(device)
print(model)

criterion = LabelSmoothingCrossEntropy()
criterion = criterion.to(device)
optimizer = optim.AdamW(model.parameters() if FINTUNE is False else model.head.parameters(), lr=0.002)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)

import time
import copy
from tqdm import tqdm
from data import get_data_loaders

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    train_loader, train_loader_len = get_data_loaders(dataset_path, 64, train=True)
    val_loader, _, val_loader_len, _ = get_data_loaders(dataset_path, 32, train=False)
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    dataset_sizes = {
        'train': train_loader_len,
        'val': val_loader_len
    }

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print("-"*10)
        
        for phase in ['train', 'val']: # We do training and validation phase per epoch
            if phase == 'train':
                model.train() # model to training mode
            else:
                model.eval() # model to evaluate
            
            running_loss = 0.0
            running_corrects = 0.0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'): # no autograd makes validation go faster
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # used for accuracy
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)    
            if phase == 'train':
                scheduler.step() # step at end of epoch
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc =  running_corrects.double() / dataset_sizes[phase]
            
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) # keep the best validation accuracy model
        print()
    time_elapsed = time.time() - since # slight error
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best Val Acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, 45)

model_path = './wswin.pth'
# example = torch.rand(1, 3, 1024, 1024)
# script_model = torch.jit.script(model, example_inputs=example)
# script_model.save(model_path)
torch.save(model_ft, model_path)