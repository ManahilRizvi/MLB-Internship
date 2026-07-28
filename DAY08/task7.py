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

    for i in range(rows):
        for j in range(columns):
            for k in range(channel):
                value=int(rgbImg[i][j][k])+brightnessVal
                if value>255:
                    value=255
                elif value<0:
                    value=0
                brighterImg[i][j][k]=value

    alphaVal=1.5
    betaVal=0
    contrastImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(columns):
            for k in range(channel):
                value=int(alphaVal*rgbImg[i][j][k]+betaVal)
                if value>255:
                    value=255
                elif value<0:
                    value=0
                contrastImg[i][j][k]=value

    kernelSharp=np.array([[0, -1, 0], 
                         [-1, 5, -1],
                         [0, -1, 0]])
    sharpenImg=np.zeros((rows, columns, channel), dtype=np.uint8)

    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        total+=rgbImg[i+x][j+y][k]*kernelSharp[x+1][y+1]
                if total>255:
                    total=255
                elif total<0:
                    total=0
                sharpenImg[i][j][k]=total

    noiseImg=rgbImg.copy()
    probability=0.03

    for i in range(rows):
        for j in range(columns):
            randomVal=np.random.rand()
            if randomVal<probability:
                for k in range(channel):
                    noiseImg[i][j][k]=0
            elif randomVal>(1-probability):
                for k in range(channel):
                    noiseImg[i][j][k]=255

    denoiseImg=np.zeros((rows, columns, channel), dtype=np.uint8)
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

                denoiseImg[i][j][k]=values[4] 

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(rgbImg)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(brighterImg)
    plt.title("Brightness Adjustment")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(contrastImg)
    plt.title("Contrast Enhancement")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(sharpenImg)
    plt.title("Sharpen Image")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(noiseImg)
    plt.title("Noise Image")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(denoiseImg)
    plt.title("Denoise Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()