import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image...")
else:
    rows, columns=img.shape
    filterImg=np.zeros((rows, columns), dtype=np.uint8)
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            #calculating sum of all 9 neighboring pixels
            total=(
                int(img[i-1][j-1])+
                int(img[i-1][j])+
                int(img[i-1][j+1])+
                int(img[i][j-1])+
                int(img[i][j])+
                int(img[i][j+1])+
                int(img[i+1][j-1])+
                int(img[i+1][j])+
                int(img[i+1][j+1])
            )
            #calculating average
            average=total//9
            filterImg[i][j]=average

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(filterImg, cmap="gray")
    plt.title("Mean Filter Image")
    plt.axis("off")
    plt.show()
#What type of noise does it reduce?
#mean filter is effective for reducing random noise. it also smooths
#small intensity variations by averaging neighboring pixels

#Why does the image become blurry?
#mean filter replaces each pixel with average of its neighboring pixels
#this thing reduces sharp intensity changes including image edges
#causing image to appear blurred