import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgbToGray(image):
    rows, columns, channel=image.shape
    gray=np.zeros((rows, columns), dtype=np.uint8)
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

def cdfCalculation(histogram):
    cdf=np.zeros(256, dtype=int)
    cdf[0]=histogram[0]
    for i in range(1, 256):
        cdf[i]=cdf[i-1]+histogram[i]
    cdf=cdf/cdf[-1]
    return cdf

def mapping(source, reference):
    map=np.zeros(256, dtype=np.uint8)
    for i in range(256):
        difference=abs(source[i]-reference[0])
        index=0
        for j in range(256):
            d1=abs(source[i]-reference[j])
            if d1<difference:
                difference=d1
                index=j

        map[i]=index
    return map

def histogramMatch(srcImg, refImg):
    srcHist=histogram(srcImg)
    refHist=histogram(refImg)

    srcCdf=cdfCalculation(srcHist)
    refCdf=cdfCalculation(refHist)

    map=mapping(srcCdf, refCdf)
    rows, columns=srcImg.shape
    match=np.zeros_like(srcImg)
    for i in range(rows):
        for j in range(columns):
            pixel=srcImg[i][j]
            match[i][j]=map[pixel]
    return match

source=cv2.imread("lowCon.jpg")
reference=cv2.imread("cat.jpeg")

if source is None or reference is None:
    print("No image...")
    exit()

srcGray=rgbToGray(source)
refGray=rgbToGray(reference)

match=histogramMatch(srcGray, refGray)
srcHist=histogram(srcGray)
refHist=histogram(refGray)
matchHist=histogram(match)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(srcGray, cmap="gray")
plt.title("Source Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(refGray, cmap="gray")
plt.title("Reference Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(match, cmap="gray")
plt.title("Matched Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(range(256), srcHist)
plt.title("Source Histogram")

plt.subplot(1, 3, 2)
plt.plot(range(256), refHist)
plt.title("Reference Histogram")

plt.subplot(1, 3, 3)
plt.plot(range(256), matchHist)
plt.title("Matched Histogram")
plt.show()

#histogram matching changes intensity distribution of source image
#matched image becomes similar to reference image and matched histogram
#follows shape of reference histogram unlike histogram equalization histogram
#matching uses another image as a reference