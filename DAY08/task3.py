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
    kernel=np.array([[0, -1, 0], 
                     [-1, 5, -1],
                     [0, -1, 0]])
    sharpenImg=np.zeros((rows, columns, channel), dtype=np.uint8)

    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        total+=rgbImg[i+x][j+y][k]*kernel[x+1][y+1]
                if total>255:
                    total=255
                elif total<0:
                    total=0
                sharpenImg[i][j][k]=total

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgbImg)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(sharpenImg)
    plt.title("Sharpened Image")
    plt.axis("off")
    plt.show()