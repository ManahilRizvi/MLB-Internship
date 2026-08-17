import numpy as np
input=np.array([2, 3, 1])
weight=np.array([0.5, 0.2, 0.8])
bias=0.5
weightSum=0
for i in range(len(input)):
    weightSum=weightSum+(input[i]*weight[i])
weightSum=weightSum+bias
print("Weighted Sum: ", weightSum)
if weightSum>0:
    output=weightSum
else:
    output=0
print("Output: ", output)