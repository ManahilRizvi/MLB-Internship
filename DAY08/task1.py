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
    brightnessVal=50
    brighterImg=rgbImg.copy()
    darkerImg=rgbImg.copy()
#brightness 
    for i in range(rows):
        for j in range(columns):
            for k in range(channel):
                value=int(rgbImg[i][j][k])+brightnessVal
                if value>255:
                    value=255
                brighterImg[i][j][k]=value
#darkness
    for i in range(rows):
            for j in range(columns):
                for k in range(channel):
                    value=int(rgbImg[i][j][k])-brightnessVal
                    if value<0:
                        value=0
                    darkerImg[i][j][k]=value

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(rgbImg)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(brighterImg)
    plt.title("Brighter Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(darkerImg)
    plt.title("Darker Image")
    plt.axis("off")
    plt.show()
