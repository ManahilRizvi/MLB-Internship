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

    kernelBlur=np.ones((3, 3))
    blurImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        total+=rgbImg[i+x][j+y][k]*kernelBlur[x+1][y+1]
                blurImg[i][j][k]=total//9

    kernelSharp=np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]])
    restoreImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        total+=blurImg[i+x][j+y][k]*kernelSharp[x+1][y+1]
                if total>255:
                    total=255
                elif total<0:
                    total=0
                restoreImg[i][j][k]=total    

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(rgbImg)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(blurImg)
    plt.title("Blurred Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(restoreImg)
    plt.title("Restored Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()