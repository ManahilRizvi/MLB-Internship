import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalization(image):
    image=np.abs(image)
    minVal=np.min(image)
    maxVal=np.max(image)
    result=np.zeros(image.shape, dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            if maxVal!=minVal:
                value=((image[i][j]-minVal)/(maxVal-minVal))*255
                result[i][j]=int(value)
    return result

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

    kernelLap=[[0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]]

    laplacianImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(grayImg[i+m][j+n])
                    value+=pixel*kernelLap[m+1][n+1]
            laplacianImg[i][j]=value

    laplacianVal=np.abs(laplacianImg)
    laplacianDisplay=normalization(laplacianVal)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(grayImg, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(laplacianDisplay, cmap="gray")
    plt.title("Laplacian: Edge Detection")
    plt.tight_layout()
    plt.show()

#laplacian operator detects edges by calculating second derivative
#of image intensity. it highlights regions where intensity changes
#rapidly. it detects using single kernel. it is more sensitive to 
#noise because it uses second order derivatives.