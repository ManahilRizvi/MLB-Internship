import numpy as np
data=np.array([[1, 20],
               [2, 42],
               [3, 46],
               [4, 85],
               [5, 17],
               [6, 82],
               [7, 24],
               [8, 66],
               [9, 88], 
               [10, 60]])

trainSet=[]
validationSet=[]
testSet=[]
for i in range(len(data)):
    if i<7:
        trainSet.append(data[i])
    elif i<9:
        validationSet.append(data[i])
    else:
        testSet.append(data[i])
trainSet=np.array(trainSet)
validationSet=np.array(validationSet)
testSet=np.array(testSet)

print("Training Set: ")
print(trainSet)
print("\nValidation Set: ")
print(validationSet)
print("\nTest Set: ")
print(testSet)

#training set: it is used to learn model. model learns patterns
#from this data and adjusts its weight.
#validation set: it is used to check and improve model performance during
#training. we see how model performs on unseen data through this.
#test set: it is used to evaluate final performance of model at 
#the end of training. Model didn't see test data during training.
#training 70% -> model learns
#validation 20% -> model is checked
#test 10% -> final evaluation