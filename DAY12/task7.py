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

def homography(point1, point2):
    A=[]
    B=[]
    for i in range(4):
        x=point1[i][0]
        y=point1[i][1]

        X=point2[i][0]
        Y=point2[i][1]

        A.append([x, y, 1, 0, 0, 0, -X*x, -X*y])
        B.append(X)

        A.append([0, 0, 0, x, y, 1, -Y*x, -Y*y])
        B.append(Y)

    A=np.array(A, dtype=np.float64)
    B=np.array(B, dtype=np.float64)

    try:
        h=np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return None
    H=np.array([[h[0], h[1], h[2]],
                [h[3], h[4], h[5]],
                [h[6], h[7], 1]])

    return H

def transformPoint(H, point):
    x=point[0]
    y=point[1]
    pointMatrix=np.array([x, y, 1], dtype=np.float64)
    result=np.dot(H, pointMatrix)
    if result[2]==0:
        return None
    newX=result[0]/result[2]
    newY=result[1]/result[2]
    return newX, newY

def errorCalculate(H, point1, point2):
    tP=transformPoint(H, point1)
    if tP is None:
        return 999999
    x=tP[0]
    y=tP[1]

    actualX=point2[0]
    actualY=point2[1]
    error=np.sqrt((x-actualX)**2+(y-actualY)**2)
    return error

def ransac(matches, keypoint1, keypoint2):
    if len(matches)<4:
        print("Can't do Ransac....")
        return [], None
    bestInliers=[]
    bestH=None
    iterations=500
    threshold=5.0
    np.random.seed(20)
    for iteration in range(iterations):
        randInd=np.random.choice(len(matches), 4, replace=False)
        point1=[]
        point2=[]
        for index in randInd:
            match=matches[index]
            index1=match[0]
            index2=match[1]
            p1=keypoint1[index1]
            p2=keypoint2[index2]

            point1.append(p1)
            point2.append(p2)

        H=homography(point1, point2)
        if H is None:
            continue
        currentInliers=[]
        for i in range(len(matches)):
            match=matches[i]
            index1=match[0]
            index2=match[1]
            point1=keypoint1[index1]
            point2=keypoint2[index2]

            error=errorCalculate(H, point1, point2)
            if error<threshold:
                currentInliers.append(i)
        if len(currentInliers)>len(bestInliers):
            bestInliers=currentInliers
            bestH=H
    return bestInliers, bestH


def matchImg(img1, img2, keypoint1, keypoint2, matches, color):
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

    for match in matches:
        indx1=match[0]
        indx2=match[1]
        x1=keypoint1[indx1][0]
        y1=keypoint1[indx1][1]

        x2=keypoint2[indx2][0]
        y2=keypoint2[indx2][1]

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
    return matchImg

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
print("Good Match before Ransac: ", len(goodMatch))

if len(goodMatch)<4:
    print("Error...")
    exit()

beforeRansac=matchImg(img1, img2, kp1, kp2, goodMatch, [255, 0, 0])
inlierIndices, H=ransac(goodMatch, kp1, kp2)
inlierMatches=[]
for index in inlierIndices:
    inlierMatches.append(goodMatch[index])

print("Inliers after Ransac: ", len(inlierMatches))
print("Outliers Removes: ", len(goodMatch)-len(inlierMatches))

afterRansac=matchImg(img1, img2, kp1, kp2, inlierMatches, [0, 255, 0])
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.imshow(beforeRansac)
plt.title("Before RANSAC")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(afterRansac)
plt.title("After RANSAC")
plt.axis("off")
plt.tight_layout()
plt.show()

#before ransac some matches may be incorrect 
#ransac fimds transformation supported by largest number of matching points
#after ransac incorrect matches are removed and only consistent
#inlier matches remain
#ransac is useful because it makes feature matching more relaibale 
#in presence of outliers