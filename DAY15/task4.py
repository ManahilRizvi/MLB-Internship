import numpy as np
def mse(actual, pred):
    error=actual-pred
    errorSqr=error*error
    return np.mean(errorSqr)

def binCrossEntropy(actual, pred):
    epsilon=1e-15
    pred=np.clip(pred, epsilon, 1-epsilon)
    loss=-(actual*np.log(pred)+(1-actual)*np.log(1-pred))
    return np.mean(loss)

def categoryCrossEntropy(actual, pred):
    epsilon=1e-15
    pred=np.clip(pred, epsilon, 1-epsilon)
    loss=-np.sum(actual*np.log(pred), axis=1)
    return np.mean(loss)

actualMse=np.array([2, 4, 6])
predMse=np.array([3, 5, 5])
mseLoss=mse(actualMse, predMse)
print("MSE Loss: ", mseLoss)

actualBce=np.array([1, 0, 1, 1])
predBce=np.array([0.9, 0.2, 0.8, 0.7])
bceLoss=binCrossEntropy(actualBce, predBce)
print("Binary Cross Entropy Loss: ", bceLoss)

actualCce=np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
predCce=np.array([[0.8, 0.1, 0.1],
                  [0.1, 0.7, 0.2],
                  [0.2, 0.2, 0.6]])
cceLoss=categoryCrossEntropy(actualCce, predCce)
print("Categorical Cross Entropy Loss: ", cceLoss)