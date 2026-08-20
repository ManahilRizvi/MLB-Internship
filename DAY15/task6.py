import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
#input data
x=np.array([0.5, 0.8])
#target
y=1.0
#hidden layer parameters
w1=np.array([[0.1, 0.2],
             [0.3, 0.4]])
b1=np.array([0.1, 0.1])
#output layer parameters
w2=np.array([0.5, 0.6])
b2=0.1
learnRate=0.1
#forward propagation
z1=np.dot(w1, x)+b1
a1=sigmoid(z1)
z2=np.dot(w2, a1)+b2
a2=sigmoid(z2)
#calculating loss
loss=0.5*(y-a2)**2
#back propagation
#output layer
da2=a2-y
dz2=da2*a2*(1-a2)
dw2=dz2*a1
db2=dz2
#hidden layer
da1=dz2*w2
dz1=da1*a1*(1-a1)
dw1=np.outer(dz1, x)
db1=dz1
#updating parameters
w2=w2-learnRate*dw2
b2=b2-learnRate*db2
w1=w1-learnRate*dw1
b1=b1-learnRate*db1

print("Prediction before update: ", a2)
print("Loss before update: ", loss)
print("Gradients: ")
print("dw1: ")
print(dw1)
print("db1: ")
print(db1)
print("dw2: ")
print(dw2)
print("db2: ")
print(db2)

print("\nParameters Updated....")
print("w1: ")
print(w1)
print("b1: ")
print(b1)
print("w2: ")
print(w2)
print("b2: ")
print(b2)