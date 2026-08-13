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

def harrisCorner(image, threshold=0.01):
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

def cornersSelection(corners, number=300):
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

def selectBestMatch(matches, number=50):
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

def canvas(baseImg, newImg, H):
    baseRows=baseImg.shape[0]
    baseCols=baseImg.shape[1]
    newRows=newImg.shape[0]
    newCols=newImg.shape[1]
    newCorner=[[0, 0],
               [newCols-1, 0],
               [newCols-1, newRows-1],
               [0, newRows-1]]
    transCorner=[]
    for point in newCorner:
        transform=homography(H, point)
        if transform is not None:
            transCorner.append(transform)
    minX=0
    minY=0
    maxX=baseCols-1
    maxY=baseRows-1

    for point in transCorner:
        if point[0]<minX:
            minX=point[0]
        if point[1]<minY:
            minY=point[1]
        if point[0]>maxX:
            maxX=point[0]
        if point[1]>maxY:
            maxX=point[1]
    minX=int(np.floor(minX))
    minY=int(np.floor(minY))
    maxX=int(np.ceil(maxX))
    maxY=int(np.ceil(maxY))

    width=maxX-minX+1
    height=maxY-minY+1
    return width, height, minX, minY

def transMatrix(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=float)

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

def blendImg(img1, img2):
    rows, columns, channel=img1.shape
    stitch=np.zeros((rows, columns, channel), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            pix1=img1[i][j]
            pix2=img2[i][j]
            exists1=False
            for c in range(channel):
                if pix1[c]!=0:
                    exists1=True
            exists2=False
            for c in range(channel):
                if pix2[c]!=0:
                    exists2=True
            if exists1 and not exists2:
                for c in range(channel):
                    stitch[i][j][c]=pix1[c]
            elif not exists1 and exists2:
                for c in range(channel):
                    stitch[i][j][c]=pix2[c]
            elif exists1 and exists2:
                for c in range(channel):
                    value=(0.5*float(pix1[c])+0.5*float(pix2[c]))
                    stitch[i][j][c]=int(value)
    return stitch

def stitchImges(baseImg, newImg):
    grayBase=rgbToGray(baseImg)
    grayNew=rgbToGray(newImg)
    cornerBase=harrisCorner(grayBase, 0.01)
    cornerNew=harrisCorner(grayNew, 0.01)
    cornerBase=cornersSelection(cornerBase, 300)
    cornerNew=cornersSelection(cornerNew, 300)

    match=featureMatch(grayBase, grayNew, cornerBase, cornerNew)
    match=selectBestMatch(match, 50)
    print("No of matches:", len(match))
    if len(match)<4:
        print("Not enough...")
        return None
    srcPoints=[]
    destPoints=[]
    for i in match:
        srcPoints.append(i[1])
        destPoints.append(i[0])
    H, inliers=ransac(srcPoints, destPoints, 1000, 3)
    print("No of Inliers: ", len(inliers))

    if H is None or len(inliers)<4:
        print("Error....")
        return None
    inlierSrc=[]
    inlierDest=[]

    for index in inliers:
        inlierSrc.append(srcPoints[index])
        inlierDest.append(destPoints[index])

    H=dlt(inlierSrc, inlierDest)
    width, height, minX, minY=canvas(baseImg, newImg, H)
    T=transMatrix(-minX, -minY)
    warpBase=warp(baseImg, T, height, width)
    newH=np.matmul(T, H)
    warpNew=warp(newImg, newH, height, width)
    panorama=blendImg(warpBase, warpNew)
    return panorama

img1=cv2.imread("book1.jpeg")
img2=cv2.imread("book2.jpg")
img3=cv2.imread("book3.jpg")

if img1 is None or img2 is None or img3 is None:
    print("No image...")
    exit()

print("Stitching image 1 and image 2")
panorama1=stitchImges(img1, img2)
if panorama1 is None:
    print("Failed...")
    exit()

print("Stitching panorama and image 3")
panorama2=stitchImges(panorama1, img3)
if panorama2 is None:
    print("Failed...")
    exit()

img1Rgb=bgrToRgb(img1)
img2Rgb=bgrToRgb(img2)
img3Rgb=bgrToRgb(img3)
panoramaRgb=bgrToRgb(panorama2)

plt.figure(figsize=(15, 6))
plt.imshow(panoramaRgb)
plt.title("Panorama")
plt.axis("off")
plt.tight_layout()
plt.show()