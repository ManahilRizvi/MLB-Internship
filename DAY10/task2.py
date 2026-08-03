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

    sobelX=[[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]

    sobelY=[[-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]]

    gxImg=np.zeros((rows, columns), dtype=np.float32)
    gyImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            sumX=0
            sumY=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(grayImg[i+m][j+n])
                    sumX+=pixel*sobelX[m+1][n+1]
                    sumY+=pixel*sobelY[m+1][n+1]
            gxImg[i][j]=sumX
            gyImg[i][j]=sumY

    magnitudeImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(rows):
        for j in range(columns):
            magnitudeImg[i][j]=np.sqrt((gxImg[i][j]**2)+(gyImg[i][j]**2))

    gxDisplay=normalization(gxImg)
    gyDisplay=normalization(gyImg)
    magDisplay=normalization(magnitudeImg)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(grayImg, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(gxDisplay, cmap="gray")
    plt.title("Sobel X")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(gyDisplay, cmap="gray")
    plt.title("Sobel Y")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(magDisplay, cmap="gray")
    plt.title("Final Edges")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

#sobel x detects vertical edges by calculating horizontal 
#intensity changes and sobel y detects horizontall edges by 
#calculating vertical intensity changes.
#gradient magnitude combines sobel x and sobel y to produce
#final edge image if compare to simple gradient method sobel gives smoother
#and stronger edges because it uses weighted kernels that reduce effevt
#of noise.