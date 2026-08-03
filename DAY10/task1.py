import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("dog.jpg")
if img is None:
    print("No image...")
else:
    rows, columns, channel=img.shape
    grayImg=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=img[i][j][0]
            green=img[i][j][1]
            red=img[i][j][2]
            grayVal=int(0.114*blue+0.587*green+0.299*red)
            if grayVal>255:
                grayVal=255
            elif grayVal<0:
                grayVal=0
            grayImg[i][j]=grayVal

    gxImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            leftPix=int(grayImg[i][j-1])
            rightPix=int(grayImg[i][j+1])
            gxImg[i][j]=rightPix-leftPix

    gyImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            topPix=int(grayImg[i-1][j])
            bottomPix=int(grayImg[i+1][j])
            gyImg[i][j]=bottomPix-topPix

    magnitudeImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(rows):
        for j in range(columns):
            magnitudeImg[i][j]=np.sqrt((gxImg[i][j]**2)+(gyImg[i][j]**2))

    directionImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(rows):
        for j in range(columns):
            directionImg[i][j]=np.arctan2(gyImg[i][j], gxImg[i][j])

    gxAbs=np.abs(gxImg)
    gxMin=np.min(gxAbs)
    gxMax=np.max(gxAbs)
    gxDisplay=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            if gxMax!=gxMin:
                value=((gxAbs[i][j]-gxMin)/(gxMax-gxMin))*255
                gxDisplay[i][j]=int(value)

    gyAbs=np.abs(gyImg)
    gyMin=np.min(gyAbs)
    gyMax=np.max(gyAbs)
    gyDisplay=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            if gyMax!=gyMin:
                value=((gyAbs[i][j]-gyMin)/(gyMax-gyMin))*255
                gyDisplay[i][j]=int(value)

    magMin=np.min(magnitudeImg)
    magMax=np.max(magnitudeImg)
    magDisplay=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            if magMax!=magMin:
                value=((magnitudeImg[i][j]-magMin)/(magMax-magMin))*255
                magDisplay[i][j]=int(value)

    dirMin=np.min(directionImg)
    dirMax=np.max(directionImg)
    dirDisplay=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            if dirMax!=dirMin:
                value=((directionImg[i][j]-dirMin)/(dirMax-dirMin))*255
                dirDisplay[i][j]=int(value)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(grayImg, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(gxDisplay, cmap="gray")
    plt.title("Gradient X")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(gyDisplay, cmap="gray")
    plt.title("Gradient Y")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(magDisplay, cmap="gray")
    plt.title("Gradient Magnitude")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(dirDisplay, cmap="gray")
    plt.title("Gradient Direction")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
