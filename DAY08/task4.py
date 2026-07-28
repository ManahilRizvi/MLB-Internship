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
    
    saltNoise=rgbImg.copy()
    pepperNoise=rgbImg.copy()
    saltPepperNoise=rgbImg.copy()
    gaussianNoise=rgbImg.copy()
    probability=0.05

    for i in range(1, rows-1):
        for j in range(1, columns-1):
            if np.random.rand()<probability:
                for k in range(channel):
                    saltNoise[i][j][k]=255

    for i in range(1, rows-1):
        for j in range(1, columns-1):
            if np.random.rand()<probability:
                for k in range(channel):
                    pepperNoise[i][j][k]=0

    for i in range(1, rows-1):
        for j in range(1, columns-1):
            randomVal=np.random.rand()
            if randomVal<probability:
                for k in range(channel):
                    saltPepperNoise[i][j][k]=0
            elif randomVal>(1-probability):
                for k in range(channel):
                    saltPepperNoise[i][j][k]=255

    meanVal=0
    sigmaVal=25
    for i in range(rows):
        for j in range(columns):
            for k in range(channel):
                noise=np.random.normal(meanVal, sigmaVal)
                value=int(rgbImg[i][j][k]+noise)
                if value>255:
                    value=255
                elif value<0:
                    value=0
                gaussianNoise[i][j][k]=value

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(rgbImg)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(saltNoise)
    plt.title("Salt Noise")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(pepperNoise)
    plt.title("Pepper Noise")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(saltPepperNoise)
    plt.title("Salt and Pepper Noise")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(gaussianNoise)
    plt.title("Gaussian Noise")
    plt.axis("off")
    plt.tight_layout()
    plt.show()