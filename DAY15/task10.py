import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
#image transform
transform=transforms.Compose([transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])])
#dataset
dataset=datasets.ImageFolder("dataset", transform=transform)
#dataloader
loader=DataLoader(dataset, batch_size=4, shuffle=True)
#loading pretrained resnet
model=resnet18(weights=ResNet18_Weights.DEFAULT)

#freezing pretrained layers
for p in model.parameters():
    p.requires_grad=False
#replacing classification head
noOfClass=len(dataset.classes)
model.fc=nn.Linear(model.fc.in_features, noOfClass)
#loss function
lossFunc=nn.CrossEntropyLoss()
#optimizer
optimizer=optim.Adam(model.fc.parameters(), lr=0.001)
#training
epochs=5
for epoch in range(epochs):
    totalLoss=0
    for images, labels in loader:
        #forward pass
        output=model(images)
        #calculating loss
        loss=lossFunc(output, labels)
        #clearing gradients
        optimizer.zero_grad()
        #back prpagation
        loss.backward()
        #updating classification head
        optimizer.step()
        totalLoss+=loss.item()

    avgLoss=totalLoss/len(loader)
    print("Epoch: ", epoch+1, "Loss: ", avgLoss)

print("Classes: ", dataset.classes)