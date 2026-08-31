import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DATASET_PATH="dataset-resized"
IMG_SIZE=128
BATCH_SIZE=32
EPOCHS=10
FINAL_LR=0.0005
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

def exploreDataset(datasetPath):
    classes=sorted(name for name in os.listdir(datasetPath)
                   if os.path.isdir(os.path.join(datasetPath, name)))
    print("\nClasses: ")
    for name in classes:
        print(name)

    print("\nImages in each class: ")
    for name in classes:
        classPath=os.path.join(datasetPath, name)
        imgs=os.listdir(classPath)
        print(f"{name}: {len(imgs)}")

    plt.figure(figsize=(12, 8))
    for i, name in enumerate(classes):
        classPath=os.path.join(datasetPath, name)
        imgs=os.listdir(classPath)
        imgPath=os.path.join(classPath, imgs[0])
        img=Image.open(imgPath)
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    print("\nImage dimensions (first image per class): ")
    for name in classes:
        classPath=os.path.join(datasetPath, name)
        imgs=os.listdir(classPath)
        imgPath=os.path.join(classPath, imgs[0])
        img=Image.open(imgPath)
        print(f"{name}: {img.size}, mode={img.mode}")
    return classes

def buildTransforms():
    trainTrans=transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(10),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    evalTrans=transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    
    visualTrans=transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(10),
                                   transforms.ToTensor(),
    ])
    return trainTrans, evalTrans, visualTrans

def getDataloaders(datasetPath, trainTrans, evalTrans):
    baseDataset=ImageFolder(root=datasetPath)
    size=len(baseDataset)
    trainSize=int(0.70*size)
    valSize=int(0.15*size)
    testSize=size-trainSize-valSize

    indices=torch.randperm(size)
    trainInd=indices[:trainSize]
    valInd=indices[trainSize:trainSize+valSize]
    testInd=indices[trainSize+valSize:]

    trainDataset=ImageFolder(root=datasetPath, transform=trainTrans)
    evalDataset=ImageFolder(root=datasetPath, transform=evalTrans)

    trainData=Subset(trainDataset, trainInd)
    valData=Subset(evalDataset, valInd)
    testData=Subset(evalDataset, testInd)

    print("\nDataset Split: ")
    print("Total: ", size)
    print("Training: ", trainSize)
    print("Validation: ", valSize)
    print("Testing: ", testSize)
    print("Class Mapping: ", baseDataset.class_to_idx)

    trainLoader=DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
    valLoader=DataLoader(valData, batch_size=BATCH_SIZE, shuffle=False)
    testLoader=DataLoader(testData, batch_size=BATCH_SIZE, shuffle=False)
    return trainLoader, valLoader, testLoader, baseDataset

def showAugmentation(datasetPath, classes, visualTrans):
    sampleClass=classes[0]
    classPath=os.path.join(datasetPath, sampleClass)
    imgs=os.listdir(classPath)
    imgPath=os.path.join(classPath, imgs[0])
    img=Image.open(imgPath).convert("RGB")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    augmented1=visualTrans(img)
    plt.subplot(1, 3, 2)
    plt.imshow(augmented1.permute(1, 2, 0))
    plt.title("Augmented 1")
    plt.axis("off")

    augmented2=visualTrans(img)
    plt.subplot(1, 3, 3)
    plt.imshow(augmented2.permute(1, 2, 0))
    plt.title("Augmented 2")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3, 16, kernel_size=3)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(2, 2)

        self.conv2=nn.Conv2d(16, 32, kernel_size=3)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(2, 2)

        self.flatten=nn.Flatten()

        self.fc1=nn.Linear(28800, 128)
        self.fcRelu=nn.ReLU()
        self.dropout=nn.Dropout(0.5)
        self.fc2=nn.Linear(128, 6)

    def forward(self, x):
        x=self.pool1(self.relu1(self.conv1(x)))
        x=self.pool2(self.relu2(self.conv2(x)))
        x=self.flatten(x)
        x=self.fcRelu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)
        return x

def trainOneEpoch(model, loader, optimizer, lossFunc):
    model.train()
    totalLoss, correct, total=0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels=imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs=model(imgs)
        loss=lossFunc(outputs, labels)
        loss.backward()
        optimizer.step()
        totalLoss+=loss.item()
        _, predicted=torch.max(outputs, 1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    return totalLoss/len(loader), 100*correct/total

def evaluate(model, loader, lossFunc, collectPred=False):
    model.eval()
    totalLoss, correct, total=0.0, 0, 0
    allLabels, allPreds=[], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels=imgs.to(device), labels.to(device)
            outputs=model(imgs)
            loss=lossFunc(outputs, labels)
            totalLoss+=loss.item()
            _, predicted=torch.max(outputs, 1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            if collectPred:
                allLabels.extend(labels.cpu().numpy())
                allPreds.extend(predicted.cpu().numpy())
    avgLoss=totalLoss/len(loader)
    accuracy=100*correct/total
    if collectPred:
        return avgLoss, accuracy, allLabels, allPreds
    return avgLoss, accuracy

def trainModel(trainLoader, valLoader, optimizerName="Adam", lr=FINAL_LR, epochs=EPOCHS, verbose=True, tag=""):
    model=SimpleCNN().to(device)
    if optimizerName=="SGD":
        optimizer=torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer=torch.optim.Adam(model.parameters(), lr=lr)

    lossFunc=nn.CrossEntropyLoss()
    history={"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        trainLoss, trainAcc=trainOneEpoch(model, trainLoader, optimizer, lossFunc)
        valLoss, valAcc=evaluate(model, valLoader, lossFunc)
        history["train_loss"].append(trainLoss)
        history["train_acc"].append(trainAcc)
        history["val_loss"].append(valLoss)
        history["val_acc"].append(valAcc)

        if verbose:
            print(f"{tag}Epoch [{epoch+1}/{epochs}] "
                  f"Loss: {trainLoss:.4f} Accuracy: {trainAcc:.2f}% "
                  f"| Val Loss: {valLoss:.4f} Val Accuracy: {valAcc:.2f}%")
    return model, history

def plotHistory(history, titlePref=""):
    epochsRange=range(1, len(history["train_loss"])+1)
    plt.figure()
    plt.plot(epochsRange, history["train_loss"], label="Training Loss")
    plt.plot(epochsRange, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{titlePref}Training and Validation Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochsRange, history["train_acc"], label="Training Accuracy")
    plt.plot(epochsRange, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{titlePref}Training and Validation Accuracy")
    plt.legend()
    plt.show()

def compareOptimizer(trainLoader, valLoader):
    print("\nOptimizer Comparison: SGD vs Adam")
    _, sgdHist=trainModel(trainLoader, valLoader, optimizerName="SGD", lr=0.001, tag="SGD")
    _, adamHist=trainModel(trainLoader, valLoader, optimizerName="Adam", lr=0.001, tag="Adam")
    epochsRange=range(1, EPOCHS+1)
    plt.figure()
    plt.plot(epochsRange, sgdHist["val_acc"], label="SGD")
    plt.plot(epochsRange, adamHist["val_acc"], label="Adam")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("SGD vs Adam (Validation Accuracy)")
    plt.legend()
    plt.show()

def learningRateExp(trainLoader, valLoader):
    print("\nLearning Rate Experiment")
    _, lr001Hist=trainModel(trainLoader, valLoader, optimizerName="Adam", lr=0.001, tag="LR 0.001")
    _, lr0005Hist=trainModel(trainLoader, valLoader, optimizerName="Adam", lr=0.0005, tag="LR 0.0005")
    epochsRange=range(1, EPOCHS+1)
    plt.figure()
    plt.plot(epochsRange, lr001Hist["val_acc"], label="LR=0.001")
    plt.plot(epochsRange, lr0005Hist["val_acc"], label="LR=0.0005")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Learning Rate Comparison")
    plt.legend()
    plt.show()

def finalEval(model, testLoader, classes):
    lossFunc=nn.CrossEntropyLoss()
    testLoss, testAcc, allLabels, allPreds=evaluate(model, testLoader, lossFunc, collectPred=True)
    print("\nTest Accuracy: ", testAcc)
    accuracy=accuracy_score(allLabels, allPreds)
    precision=precision_score(allLabels, allPreds, average="weighted", zero_division=0)
    recall=recall_score(allLabels, allPreds, average="weighted", zero_division=0)
    f1=f1_score(allLabels, allPreds, average="weighted", zero_division=0)
    print("\nEvaluation Outputs: ")
    print("Accuracy: ", accuracy*100)
    print("Precision: ", precision*100)
    print("Recall: ", recall*100)
    print("F1 Score: ", f1*100)
    cm=confusion_matrix(allLabels, allPreds)
    print("\nConfusion Matrix: ")
    print(cm)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.show()
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusionMatrix": cm}

def projectFolder(model, folderPath, evalTrans, classes):
    if not os.path.exists(folderPath):
        print(f"\n'{folderPath}' folder not found....")
        return
    imgFiles=[f for f in os.listdir(folderPath)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not imgFiles:
        print("\nNo images...")
        return 
    model.eval()
    print("\nPredictions on New Images: ")
    plt.figure(figsize=(15, 8))
    for i, filename in enumerate(imgFiles):
        imgPath=os.path.join(folderPath, filename)
        img=Image.open(imgPath).convert("RGB")
        inputImg=evalTrans(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output=model(inputImg)
            probs=F.softmax(output, dim=1)
            confidence, predicted=torch.max(probs, 1)
        predictClass=classes[predicted.item()]
        confidenceVal=confidence.item()*100
        print(f"{filename}: Predicted={predictClass} "
              f"(Confidence: {confidenceVal:.2f}%)")
        plt.subplot(2, 5, i+1)
        plt.imshow(img)
        plt.title(f"{predictClass}\n{confidenceVal:.1f}%")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    classes=exploreDataset(DATASET_PATH)
    trainTrans, evalTrans, visualTrans=buildTransforms()
    trainLoader, valLoader, testLoader, baseDataset=getDataloaders(DATASET_PATH, trainTrans, evalTrans)
    showAugmentation(DATASET_PATH, classes, visualTrans)
    compareOptimizer(trainLoader, valLoader)
    learningRateExp(trainLoader, valLoader)
    print("\nFinal Model Training")
    model, history=trainModel(trainLoader, valLoader, optimizerName="Adam", lr=FINAL_LR, tag="Final")
    plotHistory(history, titlePref="Final Model")
    finalEval(model, testLoader, baseDataset.classes)
    projectFolder(model, "examples", evalTrans, baseDataset.classes)
    torch.save(model.state_dict(), "wasteCnn.pth")
    print("\nModel Saved....")

if __name__=="__main__":
    main()
