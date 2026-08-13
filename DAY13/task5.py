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
    rows=image.shape[0]
    columns=image.shape[1]
    gray=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=float(image[i][j][0])
            green=float(image[i][j][1])
            red=float(image[i][j][2])
            value=(0.114*blue+0.587*green+0.299*red)
            gray[i][j]=int(value)
    return gray

def harrisCorner(image, threshold=0.02):
    rows=image.shape[0]
    columns=image.shape[1]
    Ix=np.zeros((rows, columns), dtype=float)
    Iy=np.zeros((rows, columns), dtype=float)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            Ix[i][j]=(float(image[i][j+1])-float(image[i-1][j]))/2
            Iy[i][j]=(float(image[i+1][j])-float(image[i-1][j]))/2
    response=np.zeros((rows, columns), dtype=float)
    k=0.04
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            sumXX=0
            sumYY=0
            sumXY=0
            for a in range(i-2, i+3):
                for b in range(j-2, j+3):
                    sumXX=sumXX+(Ix[a][b]*Ix[a][b])
                    sumYY=sumYY+(Iy[a][b]*Iy[a][b])
                    sumXY=sumXY+(Ix[a][b]*Iy[a][b])
            determinant=(sumXX*sumYY)-(sumXY*sumXY)
            trace=sumXX+sumYY
            R=determinant-k*(trace*trace)
            response[i][j]=R
    maximum=np.max(response)
    actualThre=maximum*threshold
    corners=[]
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            if response[i][j]>actualThre:
                isMax=True
                for a in range(i-1, i+2):
                    for b in range(j-1, j+2):
                        if response[a][b]>response[i][j]:
                            isMax=False
                if isMax:  
                    corners.append([j, i, response[i][j]])
    return corners

def cornersSelection(corners, number=150):
    for i in range(len(corners)):
        for j in range(i+1, len(corners)):
            if corners[j][2]>corners[i][2]:
                temp=corners[i]
                corners[i]=corners[j]
                corners[j]=temp
    select=[]
    count=0
    for point in corners:
        select.append([point[0], point[1]])
        count=count+1
        if count==number:
            break
    return select

def descriptor(image, point, patchSize=5):
    x=point[0]
    y=point[1]
    half=patchSize//2
    values=[]
    for i in range(y-half, y+half+1):
        for j in range(x-half, x+half+1):
            values.append(float(image[i][j]))
    mean=0
    for value in values:
        mean=mean+value
    mean=mean/len(values)
    total=0
    for value in values:
        diff=value-mean
        total=total+(diff*diff)
    standardDev=np.sqrt(total/len(values))
    if standardDev==0:
        return values
    normalized=[]
    for value in values:
        normalized.append((value-mean)/standardDev)
    return normalized

def euclideanDistance(des1, des2):
    total=0
    for i in range(len(des1)):
        diff=(des1[i]-des2[i])
        total=total+(diff*diff)
    return np.sqrt(total)

def featureMatch(image1, image2, corner1, corner2):
    match=[]
    for point1 in corner1:
        if(point1[0]<3 or point1[1]<3 or point1[0]>=image1.shape[1]-3 or point1[1]>=image1.shape[0]-3):
            continue
        des1=descriptor(image1, point1)
        bestDist=float("inf")
        secondBest=float("inf")
        bestPoint=None
        for point2 in corner2:
            if(point2[0]<3 or point2[1]<3 or point2[0]>=image2.shape[1]-3 or point2[1]>=image2.shape[0]-3):
                continue
            des2=descriptor(image2, point2)
            distance=euclideanDistance(des1, des2)
            if distance<bestDist:
                secondBest=bestDist
                bestDist=distance
                bestPoint=point2
            elif distance<secondBest:
                secondBest=distance
        if bestPoint is not None and secondBest!=float("inf"):
            ratio=(bestDist/secondBest)
            if ratio<0.8:
                match.append([point1, bestPoint, bestDist])
    return match

    for i in range(len(matches)):
        for j in range(i+1, len(matches)):
            if matches[j][2]<matches[i][2]:
                temp=matches[i]
                matches[i]=matches[j]
                matches[j]=temp
    select=[]
    count=0
    for match in matches:
        select.append(match)
        count=count+1
        if count==number:
            break
    return select

def matrixCreate(srcPoints, destPoints):
    A=[]
    for i in range(len(srcPoints)):
        x=srcPoints[i][0]
        y=srcPoints[i][1]
        u=destPoints[i][0]
        v=destPoints[i][1]

        row1=[-x, -y, -1,
              0, 0, 0,
              u*x, u*y, u]

        row2=[0, 0, 0,
              -x, -y, -1,
              v*x, v*y, v]

        A.append(row1)
        A.append(row2)
    return np.array(A, dtype=float)

def homography(H, point):
    x=point[0]
    y=point[1]
    homPoint=[x, y, 1]
    result=[0, 0, 0]
    for i in range(3):
        total=0
        for j in range(3):
            total=total+H[i][j]*homPoint[j]
        result[i]=total
    if result[2]==0:
        return None
    newX=result[0]/result[2]
    newY=result[1]/result[2]
    return[round(float(newX), 2), round(float(newY), 2)]

def errorCalculate(H, srcPoint, destPoint):
    transPoint=homography(H, srcPoint)
    if transPoint is None:
        return float("inf")
    diffX=transPoint[0]-destPoint[0]
    diffY=transPoint[1]-destPoint[1]
    error=np.sqrt(diffX*diffX+diffY*diffY)
    return error

def ransac(srcPoints, destPoints, iterations=100, threshold=2):
    numPoints=len(srcPoints)
    bestH=None
    bestInliers=[]
    for iteration in range(iterations):
        indices=np.random.choice(numPoints, 4, replace=False)
        sampleSrc=[]
        sampleDest=[]
        for index in indices:
            sampleSrc.append(srcPoints[index])
            sampleDest.append(destPoints[index])
        H=dlt(sampleSrc, sampleDest)
        if H is None:
            continue
        currentInliers=[]
        for i in range(numPoints):
            error=errorCalculate(H, srcPoints[i], destPoints[i])
            if error<threshold:
                currentInliers.append(i)
        if len(currentInliers)>len(bestInliers):
            bestInliers=currentInliers
            bestH=H
    return bestH, bestInliers

def dlt(srcPoints, destPoints):
    if len(srcPoints)<4:
        print("4 points are required...")
        return None
    A=matrixCreate(srcPoints, destPoints)
    U, S, Vt=np.linalg.svd(A)
    h=Vt[-1]
    H=h.reshape(3, 3)
    if H[2][2]==0:
        return None
    H=H/H[2][2]
    return H

def warp(image, H, outRows, outCols):
    inverseH=np.linalg.inv(H)
    channels=image.shape[2]
    warpImg=np.zeros((outRows, outCols, channels), dtype=image.dtype)
    for y in range(outRows):
        for x in range(outCols):
            destPoint=[x, y, 1]
            srcPoint=[0, 0, 0]
            for i in range(3):
                total=0
                for j in range(3):
                    total=total+(inverseH[i][j]*destPoint[j])
                srcPoint[i]=total
            if srcPoint[2]==0:
                continue
            srcX=srcPoint[0]/srcPoint[2]
            srcY=srcPoint[1]/srcPoint[2]
            srcX=int(round(srcX))
            srcY=int(round(srcY))
            if(srcX>=0 and srcX<image.shape[1] and srcY>=0 and srcY<image.shape[0]):
                for c in range(channels):
                    warpImg[y][x][c]=image[srcY][srcX][c]
    return warpImg

img1=cv2.imread("book1.jpeg")
img2=cv2.imread("book2.jpg")
if img1 is None or img2 is None:
    print("No image...")
    exit()

gray1=rgbToGray(img1)
gray2=rgbToGray(img2)
corner1=harrisCorner(gray1, threshold=0.02)
corner2=harrisCorner(gray2, threshold=0.02)
corner1=cornersSelection(corner1, 150)
corner2=cornersSelection(corner2, 150)
print("Corner of Image 1: ", len(corner1))
print("Corner of Image 2: ", len(corner2))

match=featureMatch(gray1, gray2, corner1, corner2)
print("No of Matches: ", len(match))
srcPoints=[]
destPoints=[]
for i in match:
    srcPoints.append(i[1])
    destPoints.append(i[0])

bestH, inliers=ransac(srcPoints, destPoints, iterations=500, threshold=5)
print("No of inliers: ", len(inliers))
if bestH is None or len(inliers)<4:
    print("Error....")
    exit()

inlierSrc=[]
inlierDest=[]
for index in inliers:
    inlierSrc.append(srcPoints[index])
    inlierDest.append(destPoints[index])

H=dlt(inlierSrc, inlierDest)
print("Estimated Homography: ")
print(H)

alignImg=warp(img2, H, img1.shape[0], img1.shape[1])
img1Rgb=bgrToRgb(img1)
img2Rgb=bgrToRgb(img2)
alignRgb=bgrToRgb(alignImg)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img1Rgb)
plt.title("Reference Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img2Rgb)
plt.title("Original Second Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(alignRgb)
plt.title("Aligned Second Image")
plt.axis("off")
plt.tight_layout()
plt.show()