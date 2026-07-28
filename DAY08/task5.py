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
    
    noiseImg=rgbImg.copy()
    probability=0.03
#noise image
    for i in range(rows):
        for j in range(columns):
            randomVal=np.random.rand()
            if randomVal<probability:
                for k in range(channel):
                    noiseImg[i][j][k]=0
            elif randomVal>(1-probability):
                for k in range(channel):
                    noiseImg[i][j][k]=255

    meanImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    gaussianImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    medianImg=np.zeros((rows, columns, channel), dtype=np.uint8)
#mean image
    kernelMean=np.ones((3, 3))
    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        total+=noiseImg[i+x][j+y][k]*kernelMean[x+1][y+1]
                meanImg[i][j][k]=total//9
#gaussian image
    kernelGau=np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]])
    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        total+=noiseImg[i+x][j+y][k]*kernelGau[x+1][y+1]
                gaussianImg[i][j][k]=total//16
#median image
    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                values=[]
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        values.append(noiseImg[i+x][j+y][k])
                a=len(values)
                for m in range(a):
                    for n in range(0, a-m-1):
                        if values[n]>values[n+1]:
                            temp=values[n]
                            values[n]=values[n+1]
                            values[n+1]=temp

                medianImg[i][j][k]=values[4]           

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(rgbImg)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(noiseImg)
    plt.title("Noise Image")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(meanImg)
    plt.title("Mean Filter")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(gaussianImg)
    plt.title("Gaussian Filter")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(medianImg)
    plt.title("Median Filter")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
