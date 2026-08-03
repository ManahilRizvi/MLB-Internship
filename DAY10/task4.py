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

    noise=np.random.normal(0, 20, (rows, columns))
    noiseImg=grayImg.astype(np.float32)+noise
    for i in range(rows):
        for j in range(columns):
            if noiseImg[i][j]>255:
                noiseImg[i][j]=255
            elif noiseImg[i][j]<0:
                noiseImg[i][j]=0
    noiseImg=noiseImg.astype(np.uint8)

    kernelGauss=[[1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]]

    gaussImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(noiseImg[i+m][j+n])
                    value+=pixel*kernelGauss[m+1][n+1]
            gaussImg[i][j]=value/16

    kernelLap=[[0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]]

    lapImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(noiseImg[i+m][j+n])
                    value+=pixel*kernelLap[m+1][n+1]
            lapImg[i][j]=value

    lapImg=np.abs(lapImg)
    lapDisplay=normalization(lapImg)

    logImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=gaussImg[i+m][j+n]
                    value+=pixel*kernelLap[m+1][n+1]
            logImg[i][j]=value

    logImg=np.abs(logImg)
    gaussDisplay=normalization(gaussImg)
    logDisplay=normalization(logImg)
    for i in range(rows):
        for j in range(columns):
            value=int(logDisplay[i][j]*3)
            if value>255:
                value=255
            logDisplay[i][j]=value

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 5, 1)
    plt.imshow(grayImg, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(noiseImg, cmap="gray")
    plt.title("Noise Image")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(lapDisplay, cmap="gray")
    plt.title("Laplacian Image")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(gaussDisplay, cmap="gray")
    plt.title("Gaussian Blur")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(logDisplay, cmap="gray")
    plt.title("LoG Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

#gaussian noise added to image. laplacian detected edges directly
#from noisy image thats why output contains more noise. LoG first smooths
#image using gaussian blur and then applied laplacian as a result
#LoG produces cleaner and smoother edges than simple laplacian operator.