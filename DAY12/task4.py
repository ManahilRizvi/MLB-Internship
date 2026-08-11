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

def sift(image, keypoints):
    gx, gy=imgGradient(image)
    descriptors=[]
    for x, y in keypoints:
        if x<8 or y<8:
            continue
        if x>=image.shape[1]-8:
            continue
        if y>=image.shape[0]-8:
            continue
        des=[]
        for i in range(y-8, y+8, 4):
            for j in range(x-8, x+8, 4):
                magnitude=[]
                orientation=[]
                for m in range(i, i+4):
                    for n in range(j, j+4):
                        dx=gx[m][n]
                        dy=gy[m][n]
                        mag=np.sqrt(dx*dx+dy*dy)
                        angle=np.degrees(np.arctan2(dy, dx))
                        if angle<0:
                            angle+=360
                        magnitude.append(mag)
                        orientation.append(angle)
                histogram=np.zeros(8)
                for k in range(len(magnitude)):
                    binInd=int(orientation[k]/45)
                    if binInd>=8:
                        binInd=7
                    histogram[binInd]+=magnitude[k]
                for value in histogram:
                    des.append(value)
        descriptors.append(des)
    return np.array(descriptors, dtype=np.float32)

def brief(image, keypoints, pairs=64):
    descriptors=[]
    np.random.seed(10)
    pointPairs=[]
    for i in range(pairs):
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

def orb(image, keypoints, pairs=64):
    descriptors=[]
    np.random.seed(10)
    pointPairs=[]
    for i in range(pairs):
        x1=np.random.randint(-8, 9)
        y1=np.random.randint(-8, 9)

        x2=np.random.randint(-8, 9)
        y2=np.random.randint(-8, 9)
        pointPairs.append((x1, y1, x2, y2))

    gx, gy=imgGradient(image)
    for x, y in keypoints:
        if x<8 or y<8:
            continue
        if x>=image.shape[1]-8:
            continue
        if y>=image.shape[0]-8:
            continue
        angle=np.arctan2(gy[y][x], gx[y][x])
        des=[]
        for x1, y1, x2, y2 in pointPairs:
            rx1=int(x1*np.cos(angle)-y1*np.sin(angle))
            ry1=int(x1*np.sin(angle)+y1*np.cos(angle))
            rx2=int(x2*np.cos(angle)-y2*np.sin(angle))
            ry2=int(x2*np.sin(angle)+y2*np.cos(angle))
            pix1=image[y+ry1][x+rx1]
            pix2=image[y+ry2][x+rx2]
            if pix1<pix2:
                des.append(1)
            else:
                des.append(0)
        descriptors.append(des)

    return np.array(descriptors, dtype=np.uint8)

def brisk(image, keypoints):
    descriptors=[]
    radius=6
    samplePoints=[]
    for angle in range(0, 360, 30):
        rad=np.radians(angle)
        x=int(radius*np.cos(rad))
        y=int(radius*np.sin(rad))
        samplePoints.append((x, y))
    for x, y in keypoints:
        if x<radius or y<radius:
            continue
        if x>=image.shape[1]-radius:
            continue
        if y>=image.shape[0]-radius:
           continue
        des=[]
        for i in range(len(samplePoints)):
            for j in range(i+1, len(samplePoints)):
                x1, y1=samplePoints[i]
                x2, y2=samplePoints[j]
                pix1=image[y+y1][x+x1]
                pix2=image[y+y2][x+x2]
                if pix1<pix2:
                    des.append(1)
                else:
                    des.append(0)
        descriptors.append(des)
    return np.array(descriptors, dtype=np.uint8)

img=cv2.imread("dog.jpg")
if img is None:
    print("No image...")
    exit()
gray=rgbToGray(img)
keypoints, cornerRes=harrisCorner(gray, threshold=0.01)
print("Keypoints: ", len(keypoints))
siftDes=sift(gray, keypoints)
briefDes=brief(gray, keypoints)
orbDes=orb(gray, keypoints)
briskDes=brisk(gray, keypoints)

print("\nSIFT")
print("No of descriptors: ", len(siftDes))
if len(siftDes)>0:
    print("Size of Descriptor: ", siftDes.shape[1])
    print("Type: Float")

print("\nBRIEF")
print("No of descriptors: ", len(briefDes))
if len(briefDes)>0:
    print("Size of Descriptor: ", briefDes.shape[1])
    print("Type: Binary")

print("\nORB")
print("No of descriptors: ", len(orbDes))
if len(orbDes)>0:
    print("Size of Descriptor: ", orbDes.shape[1])
    print("Type: Binary")

print("\nBRISK")
print("No of descriptors: ", len(briskDes))
if len(briskDes)>0:
    print("Size of Descriptor: ", briskDes.shape[1])
    print("Type: Binary") 

siftSize=0
if siftDes is not None and len(siftDes)>0:
    siftSize=len(siftDes[0])

briefSize=0
if briefDes is not None and len(briefDes)>0:
    briefSize=len(briefDes[0])

orbSize=0
if orbDes is not None and len(orbDes)>0:
    orbSize=len(orbDes[0])

briskSize=0
if briskDes is not None and len(briskDes)>0:
    briskSize=len(briskDes[0])


print("-"*70)
print("Comparison Table")
print("-"*70)
print("Method\t\tKeypoints\tDescriptor Size\tType\t\tSpeed")
print("SIFT\t\t", len(siftDes), "\t\t", siftSize, "\t\tFloat\t\tSlow")
print("BRIEF\t\t", len(briefDes), "\t\t", briefSize, "\t\tBinary\t\tVery fast")
print("ORB\t\t", len(orbDes), "\t\t", orbSize, "\t\tBinary\t\tFast")
print("BRISK\t\t", len(briskDes), "\t\t", briskSize, "\t\tBinary\t\tFast")