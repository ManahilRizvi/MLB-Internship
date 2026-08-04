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

    kernelGauss=[[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1]]

    gaussImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(grayImg[i+m][j+n])
                    value+=pixel*kernelGauss[m+1][n+1]
            gaussImg[i][j]=value/16

    kernelX=[[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]

    kernelY=[[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]

    gradientX=np.zeros((rows, columns), dtype=np.float32)
    gradientY=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            gx=0
            gy=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=gaussImg[i+m][j+n]
                    gx+=pixel*kernelX[m+1][n+1]
                    gy+=pixel*kernelY[m+1][n+1]
            gradientX[i][j]=gx
            gradientY[i][j]=gy

    magGradient=np.zeros((rows, columns), dtype=np.float32)
    for i in range(rows):
        for j in range(columns):
            gx=gradientX[i][j]
            gy=gradientY[i][j]
            magnitude=np.sqrt((gx*gx)+(gy*gy))
            magGradient[i][j]=magnitude

    directionGrad=np.zeros((rows, columns), dtype=np.float32)
    for i in range(rows):
        for j in range(columns):
            gx=gradientX[i][j]
            gy=gradientY[i][j]
            angle=np.degrees(np.arctan2(gy, gx))
            if angle<0:
                angle+=180
            directionGrad[i][j]=angle

#keeping only strongest edge pixel in gradient direction
#and remove all other neighboring pixels which makes edges thin
    nmsImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            angle=directionGrad[i][j]
            q=0
            r=0
            #comparing left and right neighbors for horizontal edges
            if (0<=angle<22.5) or (157.5<=angle<=180):
                q=magGradient[i][j+1]
                r=magGradient[i][j-1]

            #comparing diagonal neighbors for 45 degree edges
            elif 22.5<=angle<67.5:
                q=magGradient[i+1][j-1]
                r=magGradient[i-1][j+1]

            #comparing top and bottom neighbors for vertical edges
            elif 67.5<=angle<112.5:
                q=magGradient[i+1][j]
                r=magGradient[i-1][j]

            #comparing other diagonal for 135 degree edges
            elif 112.5<=angle<157.5:
                q=magGradient[i-1][j-1]
                r=magGradient[i+1][j+1]

            #keeping only strongest pixel and supress
            #all weaker pixels
            if (magGradient[i][j]>=q) and (magGradient[i][j]>=r):
                nmsImg[i][j]=magGradient[i][j]

            else:
                nmsImg[i][j]=0

#classifying pixels into strong edges, weak edges and non edge pixels
    highThr=60
    lowThre=30
    strongPix=255
    weakPix=100
    thresholdImg=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            pixel=nmsImg[i][j]

            #strong edge
            if pixel>=highThr:
                thresholdImg[i][j]=strongPix
            #weak edge
            elif pixel>=lowThre:
                thresholdImg[i][j]=weakPix
            #not an edge
            else:
                thresholdImg[i][j]=0

#checking every weak edge pixel if it is connect to strong
#edge keep it otherwise remove it
    cannyImg=thresholdImg.copy()
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            if cannyImg[i][j]==weakPix:
                connect=False
                #checking all 8 neighboring pixels
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        if cannyImg[i+m][j+n]==strongPix:
                            connect=True

                #keep connected weak edges
                if connect:
                    cannyImg[i][j]=strongPix

                #removing alone weak edges
                else:
                    cannyImg[i][j]=0

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(grayImg, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(gaussImg.astype(np.uint8), cmap="gray")
    plt.title("Gaussian Smoothing")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(magGradient, cmap="gray")
    plt.title("Gradient Magnitude")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(nmsImg, cmap="gray")
    plt.title("Non Maximum Suppression")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(thresholdImg, cmap="gray")
    plt.title("Double Thresholding")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(cannyImg, cmap="gray")
    plt.title("Canny Edge Tracking")
    plt.axis("off")
    plt.tight_layout()
    plt.show()