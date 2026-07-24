import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image...")
else:
    rows, columns=img.shape
    meanStart=time.perf_counter()
    filterImg3=np.zeros((rows, columns), dtype=np.uint8)
    #for 3x3 kernel
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            #calculating sum of all 9 neighboring pixels
            total=0
            for m in range(3):
                for n in range(3):
                    total=total+int(img[i+m-1][j+n-1])
            #calculating average
            average=total//9
            filterImg3[i][j]=average
    meanEnd=time.perf_counter()
    meanTime=meanEnd-meanStart

    #box filter
    boxStart=time.perf_counter()
    boxImg3=np.zeros((rows, columns), dtype=np.uint8)
    #for 3x3 kernel
    kernel3=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel3[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            boxImg3[i][j]=totalSum//9
    boxEnd=time.perf_counter()
    boxTime=boxEnd-boxStart
    
    #gaussian filter
    gaussianStart=time.perf_counter()
    gaussianImg3=np.zeros((rows, columns), dtype=np.uint8)
    #for 3x3 kernel
    kernel3=np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel3[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            gaussianImg3[i][j]=totalSum//16
    gaussianEnd=time.perf_counter()
    gaussianTime=gaussianEnd-gaussianStart
    
    #median filter
    medianStart=time.perf_counter()
    medianImg3=np.zeros((rows, columns), dtype=np.uint8)
    #for 3x3 
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            pixels=[]#storing neighboring pixels
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    pixels.append(pixel)
            #bubble sorting
            for x in range(len(pixels)-1):
                for y in range(len(pixels)-x-1):
                    if pixels[y]>pixels[y+1]:
                        temp=pixels[y]
                        pixels[y]=pixels[y+1]
                        pixels[y+1]=temp
            #normalizing
            medianImg3[i][j]=pixels[4]
    medianEnd=time.perf_counter()
    medianTime=medianEnd-medianStart

    print("\nExecution time")
    print("Mean filter: ", meanTime, "seconds")
    print("Box filter: ", boxTime, "seconds")
    print("Gaussian filter: ", gaussianTime, "seconds")
    print("Median filter: ", medianTime, "seconds")
            
#observations:
#mean filter was one of fastest because it only calculates average
#box filter had similar execution time because it performs weighted averaging with equal weights
#gaussian filter took slightly more time because it uses different kernel weights
#median filter was slowest because it collects wneighboring pixels and asorts them
#using bubble sort and then selects median value

#some filters are more computationally expensive because they perform more
#calculations for each pixel. mean filter is fastest because it only
#calculates average of neighboring pixels. box filter has almost
#same processing time because it also performs simple averaging. gaussian filter
#is slightly slower because it uses different weights for each neighboring pixel.
#median filter takes the most time because ir must sort neighboring
#pixel values before selecting median which requires more computation
#than the other filters.