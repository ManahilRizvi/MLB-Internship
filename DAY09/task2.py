import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image....")
else:
    noise=np.random.normal(0, 20, img.shape)
    noiseImg=img.astype(np.float32)+noise
    noiseImg=np.clip(noiseImg, 0, 255)
    noiseImg=noiseImg.astype(np.uint8)
    rows, columns=noiseImg.shape

    meanImg=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            total=0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    total+=int(noiseImg[i+x][j+y])
            meanImg[i][j]=total//9

    gaussianImg=np.zeros((rows, columns), dtype=np.uint8)
    kernel=[[1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]]
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            total=0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    pixel=int(noiseImg[i+x][j+y])
                    weight=kernel[x+1][y+1]
                    total+=pixel*weight
            gaussianImg[i][j]=total//16

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(noiseImg, cmap="gray")
    plt.title("Noise Image")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(meanImg, cmap="gray")
    plt.title("Mean Filter")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(gaussianImg, cmap="gray")
    plt.title("Gaussian Filter")
    plt.axis("off")
    plt.show()

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
    #for gaussian noise 
    #mean=0 and standard deviation=20
    noise=np.random.normal(0, 20, rgbImg.shape)
    noiseImg=rgbImg.astype(np.float32)+noise
    noiseImg=np.clip(noiseImg, 0, 255)
    noiseImg=noiseImg.astype(np.uint8)
    rows, columns, channel=noiseImg.shape

    meanImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        total+=int(noiseImg[i+x][j+y][k])
                meanImg[i][j][k]=total//9

    gaussianImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    kernel=[[1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]]
    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        pixel=int(noiseImg[i+x][j+y][k])
                        weight=kernel[x+1][y+1]
                        total+=pixel*weight
                gaussianImg[i][j][k]=total//16
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(rgbImg)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(noiseImg)
    plt.title("Noise Image")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(meanImg)
    plt.title("Mean Filter")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(gaussianImg)
    plt.title("Gaussian Filter")
    plt.axis("off")
    plt.show()
#gaussian filter performs better because:
#mean filter gives equal weight to all neighboring pixels which
#reduces noise but also blurs edges while gaussian filter gives higher 
#weight to center pixel and lower weight to surrounding pixels
#that's why gaussian filter removes gaussian noise more effectively while preserving image
#details and edges better than mean filter. 