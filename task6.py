import cv2
import numpy as np
import matplotlib.pyplot as plt
#reading image in grayscale mode
img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image")
else:
    rows, columns=img.shape#getting height and width
    
    #creating a simple 3x3 blur kernel
    #every value is 1 because we want to take average of nearby pixels
    kernel=[[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    convImage=np.zeros((rows, columns), dtype=np.uint8)

    #ignoring border pixels because 3x3 kernel can't fit there completely
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            #to store sum of neighbor pixels
            total=0
            #moving through every value of 3x3 kernel
            for x in range(3):
                for y in range(3):
                    #corresponding neighbor pixel
                    pixel=int(img[i+x-1][j+y-1])
                    #getting kernel value
                    weight=kernel[x][y]
                    total=total+(pixel*weight)
            #dividing by 9 to calculate average pixel value
            total=int(total/9)

            #keeping value inside valid grayscale range
            if total<0:
                total=0
            if total>255:
                total=255
            convImage[i][j]=total

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(convImage, cmap="gray")
    plt.title("Convulation Image")
    plt.axis("off")

    plt.show()