import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgbToGray(image):
    rows, columns, channel=image.shape
    gray=np.zeros((rows, columns),dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=image[i][j][0]
            green=image[i][j][1]
            red=image[i][j][2]
            grayVal=int(0.114*blue+0.587*green+0.299*red)
            gray[i][j]=grayVal
    return gray

def histogram(image):
    hisArr=np.zeros(256, dtype=int)
    rows, columns=image.shape
    for i in range(rows):
        for j in range(columns):
            pixel=image[i, j]
            hisArr[pixel]+=1
    return hisArr

def histEqualization(image):
    histImg=histogram(image)
    cdf=np.zeros(256, dtype=int)
    cdf[0]=histImg[0]
    for i in range(1, 256):
        cdf[i]=cdf[i-1]+histImg[i]

    cdfMin=0
    for value in cdf:
        if value!=0:
            cdfMin=value
            break
    pixelTotal=image.shape[0]*image.shape[1]

    lookup=np.zeros(256, dtype=np.uint8)
    for i in range(256):
        value=((cdf[i]-cdfMin)/(pixelTotal-cdfMin))*255
        if value<0:
            value=0
        lookup[i]=int(value)

    rows, columns=image.shape
    equalizedImg=np.zeros_like(image)
    for i in range(rows):
        for j in range(columns):
            pixel=image[i][j]
            equalizedImg[i][j]=lookup[pixel]
    return equalizedImg

img=cv2.imread("dog.jpg")
if img is None:
    print("No image...")
    exit()

grayImg=rgbToGray(img)
equImg=histEqualization(grayImg)
ogHist=histogram(grayImg)
equalizedHist=histogram(equImg)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(grayImg, cmap="gray")
plt.title("Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(equImg, cmap="gray")
plt.title("Equalized Image")
plt.axis("off")
plt.show()


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(256), ogHist)
plt.title("Original Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.plot(range(256), equalizedHist)
plt.title("Equalized Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()

#histogram equalization improves image contrast. pixel intensitites
#spread over wider range
#dark regions become more visible and equalized histogram is more
#uniformly distributed