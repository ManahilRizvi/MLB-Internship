import cv2
import numpy as np
import matplotlib.pyplot as plt

def euclideanDist(descriptor1, descriptor2):
    total=0
    for i in range(len(descriptor1)):
        difference=descriptor1[i]-descriptor2[i]
        total=total+(difference*difference)
    distance=np.sqrt(total)
    return distance

def hammingDist(descriptor1, descriptor2):
    distance=0
    for i in range(len(descriptor1)):
        if descriptor1[i]!=descriptor2[i]:
            distance=distance+1
    return distance

descriptor1=np.array([1.0, 2.0, 3.0, 4.0])
descriptor2=np.array([2.0, 4.0, 6.0, 8.0])
euclidean=euclideanDist(descriptor1, descriptor2)
print("Euclidean Distance...")
print("Descriptor1: ", descriptor1)
print("Descriptor2: ", descriptor2)
print("Distance: ", euclidean)

binary1=np.array([1, 0, 1, 1, 0, 0, 1, 0])
binary2=np.array([1, 1, 1, 0, 0, 1, 1, 0])
hamming=hammingDist(binary1, binary2)
print("Hamming Distance...")
print("Binary Descriptor1: ", binary1)
print("Binary Descriptor2: ", binary2)
print("Distance: ", hamming)

#euclidean distance is used for floating point descriptors and it 
#measures straight line distance between two descriptors
#hamming diatsnce is used for binary descriptors and it counts
#how many positions are different between two
#binary descriptors