import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image...")
else:
    rows, columns=img.shape
    filterImg=np.zeros((rows, columns), dtype=np.uint8)
    #box filter kernel
    kernel=np.array([[1, 2, 1,], [2, 4, 2,], [1, 2, 1,]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing by dividing with sum of kernel values (16)
            filterImg[i][j]=totalSum//16

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(filterImg, cmap="gray")
    plt.title("Guassian Filter Image")
    plt.axis("off")
    plt.show()
#Why Gaussian preserves edges better than the Mean Filter.
#it assigns higher weight to center pixel and lower weights to 
#surrounding pixels this means nearby pixels influence output 
#more than distant pixels reducing unnecessary smoothing around edges

#Compare the results with the Mean Filter.
#mean filter                  |    guassian filter
#1. equal weights for all     |    1. different weights for 
#pixels                       |    each pixel
#2. uses simple average       |    2. uses weighted average
#3. produces more blur        |    3. produces less blur
#4. blurs edges more          |    4. preserves edges better
#5. simpler calculation       |    5. expensive calculation than mean filter