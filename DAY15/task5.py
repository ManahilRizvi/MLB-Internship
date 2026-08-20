import numpy as np
import matplotlib.pyplot as plt
#input data
x=np.array([1, 3, 5, 7, 9], dtype=float)
#target values
y=np.array([2, 4, 6, 8, 10], dtype=float)
weight=0.0
bias=0.0
learnRate=0.01
#no of iterations
epochs=100
#store loss values
lossVals=[]
for epoch in range(epochs):
    #forward pass
    pred=weight*x+bias
    #calculating mse loss
    error=pred-y
    loss=np.mean(error**2)
    #cslculating gradients
    dw=(2/len(x))*np.sum(x*error)
    db=(2/len(x))*np.sum(error)
    #updating weight and bias
    weight=weight-learnRate*dw
    bias=bias-learnRate*db
    #storing loss
    lossVals.append(loss)
    #progress printing
    if epoch%10==0:
        print("Epoch: ", epoch, ", Loss: ", loss, ", Weight: ", weight, ", Bias: ", bias)
print("\nFinal Weight: ", weight)
print("Final Bias: ", bias)

plt.figure()
plt.plot(lossVals)
plt.title("Loss during Gradient Descent")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid()
plt.show()