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

def shiTomasi(image, threshold=0.01):
    gx, gy=imgGradient(image)
    rows, columns=image.shape
    cornerRes=np.zeros((rows, columns), dtype=np.float32)
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
            discriminant=(trace*trace-4*determinant)
            if discriminant<0:
                discriminant=0
            sqrtVal=np.sqrt(discriminant)
            lambda1=(trace+sqrtVal)/2
            lambda2=(trace-sqrtVal)/2
            if lambda1<lambda2:
                response=lambda1
            else:
                response=lambda2
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
    
def drawKeypoints(image, keypoints):
    result=image.copy()
    for x, y in keypoints:
        for i in range(max(0, y-3), min(result.shape[0], y+4)):
            for j in range(max(0, x-3), min(result.shape[1], x+4)):
                result[i][j]=[0, 0, 255]
    return result

img=cv2.imread("dog.jpg")
if img is None:
    print("No image...")
    exit()

gray=rgbToGray(img)
threshold1=0.01
threshold2=0.05
threshold3=0.10

h1, hRes1=harrisCorner(gray, threshold1)
h2, hRes2=harrisCorner(gray, threshold2)
h3, hRes3=harrisCorner(gray, threshold3)

s1, sRes1=shiTomasi(gray, threshold1)
s2, sRes2=shiTomasi(gray, threshold2)
s3, sRes3=shiTomasi(gray, threshold3)

harrisImg1=drawKeypoints(img, h1)
harrisImg2=drawKeypoints(img, h2)
harrisImg3=drawKeypoints(img, h3)

shiImg1=drawKeypoints(img, s1)
shiImg2=drawKeypoints(img, s2)
shiImg3=drawKeypoints(img, s3)

ogRgb=bgrToRgb(img)
harrisRgb1=bgrToRgb(harrisImg1)
harrisRgb2=bgrToRgb(harrisImg2)
harrisRgb3=bgrToRgb(harrisImg3)

shiRgb1=bgrToRgb(shiImg1)
shiRgb2=bgrToRgb(shiImg2)
shiRgb3=bgrToRgb(shiImg3)

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(ogRgb)
plt.title("Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(harrisRgb1)
plt.title("Harris: Threshold=0.01")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(harrisRgb2)
plt.title("Harris: Threshold=0.05")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(harrisRgb3)
plt.title("Harris: Threshold=0.10")
plt.axis("off")
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(ogRgb)
plt.title("Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(shiRgb1)
plt.title("Shi-Tomasi: Threshold=0.01")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(shiRgb2)
plt.title("Shi-Tomasi: Threshold=0.05")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(shiRgb3)
plt.title("Shi-Tomasi: Threshold=0.10")
plt.axis("off")
plt.show()

#lower threshold detects more corners and higher threshold 
#detects fewer and stronger corners
#harris uses corner response based on determinant and trace while shi tomasi uses smaller
#eigenvalue of structure matrix
#shi tomasi focuses on points that have strong intensity