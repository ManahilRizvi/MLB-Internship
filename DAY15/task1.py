import numpy as np
def activationFunc(value):
    if value>=0:
        return 1
    else:
        return 0
#data for training
inputs=np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])
targets=np.array([0, 0, 0, 1])#target values for AND gate
weights=np.zeros(2)
bias=0.0
learnRate=0.1
epochs=10
for epoch in range(epochs):
    totalErr=0
    for i in range(len(inputs)):
        x=inputs[i]
        target=targets[i]
        #calcluating weighted sum
        weightSum=np.dot(x, weights)+bias
        #making prediction
        prediction=activationFunc(weightSum)
        #calculating error
        error=target-prediction
        #updating weights and biases
        weights=weights+learnRate*error*x
        bias=bias+learnRate*error
        if error!=0:
            totalErr+=1
    print("Epoch: ", epoch+1)
    print("Weights: ", weights)
    print("Bias: ", bias)
    print("Errors: ", totalErr)
    print()
#testing 
print("Predictions: ")
for i in range(len(inputs)):
    x=inputs[i]
    weightSum=np.dot(x, weights)+bias
    prediction=activationFunc(weightSum)
    print(x, "=", prediction)
