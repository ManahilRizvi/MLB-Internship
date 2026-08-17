import numpy as np
import matplotlib.pyplot as plt

input=np.array([1, 3, 5, 6, 9], dtype=float)
target=np.array([2, 4, 6, 8, 10], dtype=float)
weight=0.0
bias=0.0
learnRate=0.01
epochs=100
loss=[]
for e in range(epochs):
    prediction=np.zeros(len(input))
    for i in range(len(input)):
        prediction[i]=input[i]*weight+bias
    total=0
    for i in range(len(target)):
        diff=prediction[i]-target[i]
        total=total+(diff*diff)
    ls=total/len(target)
    loss.append(ls)

    weightGrad=0
    biasGrad=0
    for i in range(len(input)):
        diff=prediction[i]-target[i]
        weightGrad=weightGrad+(diff*input[i])
        biasGrad=biasGrad+diff
    weightGrad=(2/len(input))*weightGrad
    biasGrad=(2/len(input))*biasGrad
    weight=weight-learnRate*weightGrad
    bias=bias-learnRate*biasGrad
    if (e+1)%10==0:
        print("Epoch: ", e+1, "Loss: ", ls)

print("Final Weight: ", weight)
print("Final Bias: ", bias)
print("\nInput         Target         Prediction") 
for i in range(len(input)):
    prediction=input[i]*weight+bias
    print(input[i], "         ", target[i], "         ", prediction)

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs+1), loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss VS Epochs")
plt.grid(True)
plt.show()