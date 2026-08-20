import torch
import torch.nn as nn
import torch.optim as optim
#neyral network
class simpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(2, 4)
        self.layer2=nn.Linear(4, 1)
    def forward(self, x):
        x=self.layer1(x)
        x=torch.relu(x)
        x=self.layer2(x)
        return x
#input data
x=torch.tensor([[1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0]])
#target values
y=torch.tensor([[3.0], [5.0], [7.0], [9.0]])
#creating model
model=simpleNN()
#loss function
lossFunc=nn.MSELoss()
#optimizer
optimizer=optim.SGD(model.parameters(), lr=0.01)
#training
for epoch in range(1000):
    #forward pass
    pred=model(x)
    #calculating loss
    loss=lossFunc(pred, y)
    #clearing old gradients
    optimizer.zero_grad()
    #backpropagation
    loss.backward()
    #updating parameters
    optimizer.step()
    if epoch%100==0:
        print("Epoch: ", epoch, ", Loss: ", loss.item())
#testing model
testData=torch.tensor([[5.0, 6.0]])
pred=model(testData)
print("Prediction for [5, 6]: ", pred.item())