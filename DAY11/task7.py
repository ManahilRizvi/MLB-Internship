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
    return cdf

def normalizeCdf(histogram):
    cdf=np.zeros(256, dtype=float)
    cdf[0]=histogram[0]
    for i in range(1, 256):
        cdf[i]=cdf[i-1]+histogram[i]
    cdf=cdf/cdf[-1]
    return cdf

def histEqualization(image):
    histImg=histogram(image)
    cdf=cdfCalculation(histImg)

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
        if value>255:
            value=255
        lookup[i]=int(value)

    rows, columns=image.shape
    equalizedImg=np.zeros_like(image)
    for i in range(rows):
        for j in range(columns):
            pixel=image[i][j]
            equalizedImg[i][j]=lookup[pixel]
    return equalizedImg

def clipHist(histogram, clipLimit):
    histClipped=histogram.copy()
    extraPix=0
    for i in range(256):
        if histClipped[i]>clipLimit:
            extraPix+=histClipped[i]-clipLimit
            histClipped[i]=clipLimit
    return histClipped, extraPix

def pixelsRedistribution(histogram, extraPix):
    value=extraPix//256
    remainder=extraPix%256
    for i in range(256):
        histogram[i]+=value

    for i in range(remainder):
        histogram[i]+=1
    return histogram


def tileEqualize(tile, clipLimit):
    hist=histogram(tile)
    histClipped, extraPix=clipHist(hist, clipLimit)
    histClipped=pixelsRedistribution(histClipped, extraPix)
    cdf=cdfCalculation(histClipped)
    cdfMin=0
    for value in cdf:
        if value!=0:
            cdfMin=value
            break
    totalPix=tile.shape[0]*tile.shape[1]
    lookupTable=np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if totalPix==cdfMin:
            lookupTable[i]=i
        else:
            newVal=((cdf[i]-cdfMin)/(totalPix-cdfMin))*255
            if newVal<0:
                newVal=0
            if newVal>255:
                newVal=255
            lookupTable[i]=int(newVal)

    rows, columns=tile.shape
    output=np.zeros_like(tile)
    for i in range(rows):
        for j in range(columns):
            pixel=tile[i][j]
            output[i][j]=lookupTable[pixel]
    return output

def clahe(image, tileSize=8, clipLimit=40):
    rows, columns=image.shape
    output=np.zeros_like(image)
    for i in range(0, rows, tileSize):
        for j in range(0, columns, tileSize):
            rowEnd=i+tileSize
            colEnd=j+tileSize
            if rowEnd>rows:
                rowEnd=rows
            if colEnd>columns:
                colEnd=columns
            tileRow=rowEnd-i
            tileCol=colEnd-j
            tile=np.zeros((tileRow, tileCol), dtype=np.uint8)
            for m in range(tileRow):
                for n in range(tileCol):
                    tile[m][n]=image[i+m][j+n]
            equaTile=tileEqualize(tile, clipLimit)
            for r in range(tileRow):
                for c in range(tileCol):
                    output[i+r][j+c]=equaTile[r][c]
    return output

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

    srcCdf=normalizeCdf(srcHist)
    refCdf=normalizeCdf(refHist)

    map=mapping(srcCdf, refCdf)
    rows, columns=srcImg.shape
    match=np.zeros_like(srcImg)
    for i in range(rows):
        for j in range(columns):
            pixel=srcImg[i][j]
            match[i][j]=map[pixel]
    return match

img1=cv2.imread("lowCon.jpg")
img2=cv2.imread("lowImg2.jpeg")
reference=cv2.imread("cat.jpeg")

if img1 is None or img2 is None or reference is None:
    print("No image...")
    exit()

gray1=rgbToGray(img1)
gray2=rgbToGray(img2)
refGray=rgbToGray(reference)

equalize1=histEqualization(gray1)
equalize2=histEqualization(gray2)

clahe1=clahe(gray1, tileSize=8, clipLimit=40)
clahe2=clahe(gray2, tileSize=8, clipLimit=40)

match1=histogramMatch(gray1, refGray)
match2=histogramMatch(gray2, refGray)

ogHist1=histogram(gray1)
eqHist1=histogram(equalize1)
claheHist1=histogram(clahe1)
matchHist1=histogram(match1)

ogHist2=histogram(gray2)
eqHist2=histogram(equalize2)
claheHist2=histogram(clahe2)
matchHist2=histogram(match2)
refHist=histogram(refGray)

plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.imshow(gray1, cmap="gray")
plt.title("Image 1")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(equalize1, cmap="gray")
plt.title("Histogram Equalization 1")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(clahe1, cmap="gray")
plt.title("CLAHE 1")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(match1, cmap="gray")
plt.title("Histogram Matching 1")
plt.axis("off")
plt.show()

plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.plot(range(256), ogHist1)
plt.title("Image 1")

plt.subplot(1, 4, 2)
plt.plot(range(256), eqHist1)
plt.title("Histogram Equalization 1")

plt.subplot(1, 4, 3)
plt.plot(range(256), claheHist1)
plt.title("CLAHE 1")

plt.subplot(1, 4, 4)
plt.plot(range(256), matchHist1)
plt.title("Histogram Matching 1")
plt.show()
#--------------------------------------------------
plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.imshow(gray2, cmap="gray")
plt.title("Image 2")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(equalize2, cmap="gray")
plt.title("Histogram Equalization 2")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(clahe2, cmap="gray")
plt.title("CLAHE 2")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(match2, cmap="gray")
plt.title("Histogram Matching 2")
plt.axis("off")
plt.show()

plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.plot(range(256), ogHist2)
plt.title("Image 2")

plt.subplot(1, 4, 2)
plt.plot(range(256), eqHist2)
plt.title("Histogram Equalization 2")

plt.subplot(1, 4, 3)
plt.plot(range(256), claheHist2)
plt.title("CLAHE 2")

plt.subplot(1, 4, 4)
plt.plot(range(256), matchHist2)
plt.title("Histogram Matching 2")
plt.show()

#histogram equalization improves overall image contrast
#histogram matching changes source image according to reference image
#histogram equalization is suitable for general contrast enhancement
#histogram matching is useful when appearance of one image needs to resemble another image
#clahe improves local contrast br processing small image tiles
#clahe preserves image details better than histogram equalization
#clahe is more suitable for low contrast images because it enhances local details