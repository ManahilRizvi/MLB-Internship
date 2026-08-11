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

def drawKeypoints(image, keypoints):
    result=image.copy()
    for x, y in keypoints:
        for i in range(max(0, y-3), min(result.shape[0], y+4)):
            for j in range(max(0, x-3), min(result.shape[1], x+4)):
                result[i][j]=[0, 0, 255]
    return result

img1=cv2.imread("dog.jpg")
img2=cv2.imread("mouse.jpeg")
img3=cv2.imread("orange-flower.jpg")

if img1 is None or img2 is None or img3 is None:
    print("No image...")
    exit()

gray1=rgbToGray(img1)
gray2=rgbToGray(img2)
gray3=rgbToGray(img3)

kp1, res1=harrisCorner(gray1)
kp2, res2=harrisCorner(gray2)
kp3, res3=harrisCorner(gray3)

result1=drawKeypoints(img1, kp1)
result2=drawKeypoints(img2, kp2)
result3=drawKeypoints(img3, kp3)

rgb1=bgrToRgb(result1)
rgb2=bgrToRgb(result2)
rgb3=bgrToRgb(result3)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(rgb1)
plt.title("Image 1: Keypoints")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(rgb2)
plt.title("Image 2: Keypoints")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(rgb3)
plt.title("Image 3: Keypoints")
plt.axis("off")
plt.show()

#corners produce strong intensity changes in mutliple directions and
#edge produces intensity changes mainly in one direction
#flat regions have very little intensity changes
#that's why strong corners are detected as important keypoints
