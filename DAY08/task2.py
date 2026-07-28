import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("dog.jpg")
if img is None:
    print("No image...")
else:
    rows, columns, channel=img.shape
    rgbImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=img[i][j][0]
            green=img[i][j][1]
            red=img[i][j][2]

            rgbImg[i][j][0]=red
            rgbImg[i][j][1]=green
            rgbImg[i][j][2]=blue
    alpha1Val=0.5
    alpha2Val=1.5
    betaVal=0
    lowContrastImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    highContrastImg=np.zeros((rows, columns, channel), dtype=np.uint8)
#low contrast
    for i in range(rows):
        for j in range(columns):
            for k in range(channel):
                value=int(alpha1Val*rgbImg[i][j][k]+betaVal)
                if value>255:
                    value=255
                elif value<0:
                    value=0
                lowContrastImg[i][j][k]=value
#high contrast 
    for i in range(rows):
            for j in range(columns):
                for k in range(channel):
                    value=int(alpha2Val*rgbImg[i][j][k]+betaVal)
                    if value>255:
                        value=255
                    elif value<0:
                        value=0
                    highContrastImg[i][j][k]=value

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(rgbImg)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(lowContrastImg)
    plt.title("Low Contrast Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(highContrastImg)
    plt.title("High Contrast Image")
    plt.axis("off")
    plt.show()