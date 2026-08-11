import cv2
import numpy as np
import matplotlib.pyplot as plt

def bgrToRgb(image):
    rows, columns, channel=image.shape
    rgb=np.zeros_like(image)
    for i in range(rows):
        for j in range(columns):
            blue=image[i][j][0]
            green=image[i][j][1]
            red=image[i][j][2]
            rgb[i][j][0]=red
            rgb[i][j][1]=green
            rgb[i][j][2]=blue
    return rgb

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

def imgGradient(image):
    rows, columns=image.shape
    gx=np.zeros((rows, columns), dtype=np.float32)
    gy=np.zeros((rows, columns), dtype=np.float32)

    sobelX=np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
    sobelY=np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])

    for i in range(1, rows-1):
        for j in range(1, columns-1):
            region=np.zeros((3, 3), dtype=np.float32)
            for m in range(3):
                for n in range(3):
                    region[m][n]=image[i-1+m][j-1+n]
            gx[i][j]=np.sum(region*sobelX)
            gy[i][j]=np.sum(region*sobelY)

    return gx, gy

def harrisCorner(image, threshold=0.01):
    gx, gy=imgGradient(image)
    rows, columns=image.shape
    cornerRes=np.zeros((rows, columns), dtype=np.float32)
    k=0.04
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            gxReg=np.zeros((3, 3), dtype=np.float32)
            gyReg=np.zeros((3, 3), dtype=np.float32)
            for m in range(3):
                for n in range(3):
                    gxReg[m][n]=gx[i-1+m][j-1+n]
                    gyReg[m][n]=gy[i-1+m][j-1+n]
            sumGx=np.sum(gxReg*gxReg)
            sumGy=np.sum(gyReg*gyReg)
            sumGxy=np.sum(gxReg*gyReg)
            determinant=(sumGx*sumGy-sumGxy*sumGxy)
            trace=sumGx+sumGy
            response=determinant-k*(trace*trace)
            cornerRes[i][j]=response
    maxRes=np.max(cornerRes)
    keypoints=[]
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            if cornerRes[i][j]>threshold*maxRes:
                neighborhood=np.zeros((3, 3), dtype=np.float32)
                for m in range(3):
                    for n in range(3):
                        neighborhood[m][n]=cornerRes[i-1+m][j-1+n]
                if cornerRes[i][j]==np.max(neighborhood):
                    keypoints.append((j, i))
    return keypoints, cornerRes

def descriptors(image, keypoints):
    descriptors=[]
    np.random.seed(10)
    pointPairs=[]
    for i in range(64):
        x1=np.random.randint(-8, 9)
        y1=np.random.randint(-8, 9)

        x2=np.random.randint(-8, 9)
        y2=np.random.randint(-8, 9)
        pointPairs.append((x1, y1, x2, y2))

    for x, y in keypoints:
        if x<8 or y<8:
            continue
        if x>=image.shape[1]-8:
            continue
        if y>=image.shape[0]-8:
            continue
        des=[]
        for x1, y1, x2, y2 in pointPairs:
            pix1=image[y+y1][x+x1]
            pix2=image[y+y2][x+x2]
            if pix1<pix2:
                des.append(1)
            else:
                des.append(0)
        descriptors.append(des)

    return np.array(descriptors, dtype=np.uint8)

def hammingDist(descriptor1, descriptor2):
    distance=0
    for i in range(len(descriptor1)):
        if descriptor1[i]!=descriptor2[i]:
            distance=distance+1
    return distance

def featuresMatching(des1, des2):
    knnMatch=[]
    for i in range(len(des1)):
        distances=[]
        for j in range(len(des2)):
            d=hammingDist(des1[i], des2[j])
            distances.append((d, j))
        for m in range(len(distances)):
            for n in range(m+1, len(distances)):
                if distances[n][0]<distances[m][0]:
                    temp=distances[m]
                    distances[m]=distances[n]
                    distances[n]=temp
        if len(distances)>=2:
            best=distances[0]
            second=distances[1]
            knnMatch.append((i, best, second))
    goodMatch=[]
    ratio=0.75
    for match in knnMatch:
        desInd=match[0]
        bestDist=match[1][0]
        bestInd=match[1][1]
        secondDist=match[2][0]
        if bestDist<ratio*secondDist:
            goodMatch.append((desInd, bestInd, bestDist))

    return goodMatch

img1=cv2.imread("book1.jpg")
img2=cv2.imread("book2.jpg")
if img1 is None or img2 is None:
    print("No image...")
    exit()

gray1=rgbToGray(img1)
gray2=rgbToGray(img2)
kp1, res1=harrisCorner(gray1, threshold=0.001)
kp2, res2=harrisCorner(gray2, threshold=0.001)
print("Image 1 Keypoints: ", len(kp1))
print("Image 2 Keypoints: ", len(kp2))
descriptor1=descriptors(gray1, kp1)
descriptor2=descriptors(gray2, kp2)
print("Image 1 Descriptors: ", len(descriptor1))
print("Image 2 Descriptors: ", len(descriptor2))
if len(descriptor1)==0 or len(descriptor2)==0:
    print("Error...")
    exit()

goodMatch=featuresMatching(descriptor1, descriptor2)
print("Good Match: ", len(goodMatch))
rgb1=bgrToRgb(img1)
rgb2=bgrToRgb(img2)
height1=rgb1.shape[0]
width1=rgb1.shape[1]

height2=rgb2.shape[0]
width2=rgb2.shape[1]
if height1>height2:
    height=height1
else:
    height=height2
width=width1+width2

matchImg=np.zeros((height, width, 3), dtype=np.uint8)
for i in range(height1):
    for j in range(width1):
        matchImg[i][j][0]=rgb1[i][j][0]
        matchImg[i][j][1]=rgb1[i][j][1]
        matchImg[i][j][2]=rgb1[i][j][2]

for i in range(height2):
    for j in range(width2):
        matchImg[i][width1+j][0]=rgb2[i][j][0]
        matchImg[i][width1+j][1]=rgb2[i][j][1]
        matchImg[i][width1+j][2]=rgb2[i][j][2]

for match in goodMatch:
    indx1=match[0]
    indx2=match[1]
    x1=kp1[indx1][0]
    y1=kp1[indx1][1]

    x2=kp2[indx2][0]
    y2=kp2[indx2][1]

    x2=x2+width1
    dx=x2-x1
    dy=y2-y1
    steps=max(abs(dx), abs(dy))
    if steps==0:
        steps=1
    for step in range(steps+1):
        x=int(x1+(dx*step/steps))
        y=int(y1+(dy*step/steps))
        if 0<=x<width and 0<=y<height:
            matchImg[y][x]=[255, 0, 0]

    for py in range(max(0, y1-3), min(height, y1+4)):
        for px in range(max(0, x1-3), min(width, x1+4)):
            matchImg[py][px]=[0, 255, 0]

    for py in range(max(0, y2-3), min(height, y2+4)):
            for px in range(max(0, x2-3), min(width, x2+4)):
                matchImg[py][px]=[0, 255, 0]
plt.figure(figsize=(15, 8))
plt.imshow(matchImg)
plt.title("Good Feature Matches")
plt.axis("off")
plt.show()