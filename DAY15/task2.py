import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x)/np.exp(x)+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leakyRelu(x, alpha=0.01):
    result=np.zeros_like(x)
    for i in range(len(x)):
        if x[i]>0:
            result[i]=x[i]
        else:
            result[i]=alpha*x[i]
    return result

def softmax(x):
    shiftX=x-np.max(x)
    expVal=np.exp(shiftX)
    return expVal/np.sum(expVal)
#input values
x=np.linspace(-10, 10, 400)
#calculating activation outputs
sigmoidOut=sigmoid(x)
tanhOut=tanh(x)
reluOut=relu(x)
leakyReluOut=leakyRelu(x)

softmaxInp=np.array([-2, -1, 0, 1, 2])
softmaxOut=softmax(softmaxInp)
print("Softmax Input: ", softmaxInp)
print("Softmax Output: ", softmaxOut)
print("Softmax Sum: ", np.sum(softmaxOut))

plt.figure()
plt.plot(x, sigmoidOut)
plt.title("Sigmoid")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

plt.figure()
plt.plot(x, tanhOut)
plt.title("Tanh")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

plt.figure()
plt.plot(x, reluOut)
plt.title("ReLU")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

plt.figure()
plt.plot(x, leakyReluOut)
plt.title("Leaky ReLU")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

plt.figure()
plt.plot(softmaxInp, softmaxOut, marker="o")
plt.title("Softmax")
plt.xlabel("Input")
plt.ylabel("Probability")
plt.grid()
plt.show()