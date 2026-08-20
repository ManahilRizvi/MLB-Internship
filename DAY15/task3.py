import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

#input values
x=np.array([1.0, 2.0, 3.0])
#hidden layer weights
w1=np.array([[0.1, 0.2, 0.3],
             [0.4, 0.5, 0.6],
             [0.7, 0.8, 0.9]])
#hidden layer bias
b1=np.array([0.1, 0.1, 0.1])
#output layer weights
w2=np.array([[0.2, 0.4, 0.6]])
#output layer bias
b2=np.array([0.1])

#forward propagation
#hidden layer weighted sum
z1=np.dot(w1, x)+b1
#hidden layer activation
a1=relu(z1)
#output layer weighted sum
z2=np.dot(w2, a1)+b2
#output
a2=sigmoid(z2)
print("Input: ")
print(x)
print("\nHidden Layer Weighted Sum: ")
print(z1)
print("\nHidden Layer Output: ")
print(a1)
print("\nOutput Layer Weighted Sum: ")
print(z2)
print("\nPrediction: ")
print(a2)