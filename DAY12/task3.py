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

def fastDetector(image, threshold):
    rows, columns=image.shape
    keypoints=[]
    circle=[(-3, 0), (-3, 1), (-2, 2), (-1, 3),
            (0, 3), (1, 3), (2, 2), (3, 1),
            (3, 0), (3, -1), (2, -2), (1, -3),
            (0, -3), (-1, -3), (-2, -2), (-3, -1)]
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            center=int(image[i][j])
            brighter=[]
            darker=[]
            for dy, dx in circle:
                pixel=int(image[i+dy][j+dx])
                if pixel>center+threshold:
                    brighter.append(1)
                else:
                    brighter.append(0)
                if pixel<center-threshold:
                    darker.append(1)
                else:
                    darker.append(0)
            brightCheck=brighter+brighter[:11]
            darkCheck=darker+darker[:11]

            brightCount=0
            darkCount=0
            cornerCheck=False
            for k in range(len(brightCheck)):
                if brightCheck[k]==1:
                    brightCount+=1
                else:
                    brightCount=0

                if brightCount>=12:
                    cornerCheck=True
                    break
                if darkCheck[k]==1:
                    darkCount+=1
                else:
                    darkCount=0

                if darkCount>=12:
                    cornerCheck=True
                    break
            if cornerCheck:
                keypoints.append((j, i))
    return keypoints

def drawKeypoints(image, keypoints):
    result=image.copy()
    for x, y in keypoints:
        for i in range(max(0, y-3), min(result.shape[0], y+4)):
            for j in range(max(0, x-3), min(result.shape[1], x+4)):
                result[i][j]=[0, 0, 255]
    return result


img=cv2.imread("dog.jpg")
if img is None:
    print("No image....")
    exit()

gray=rgbToGray(img)
kp10=fastDetector(gray, 10)
kp30=fastDetector(gray, 30)
kp50=fastDetector(gray, 50)

img10=drawKeypoints(img, kp10)
img30=drawKeypoints(img, kp30)
img50=drawKeypoints(img, kp50)

rgb10=bgrToRgb(img10)
rgb30=bgrToRgb(img30)
rgb50=bgrToRgb(img50)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(rgb10)
plt.title("FAST: Threshold=10")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(rgb30)
plt.title("FAST: Threshold=30")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(rgb50)
plt.title("FAST: Threshold=50")
plt.axis("off")
plt.show()

#lower threshold detects more keypoints and higher threshold
#detects fewer but stronger keypoints. fast checsks intensity
#changes around circular neighborhood and fast is fatser than methods
#that calculate complex cprner responses
#strong intensity changes around circle are detected as corners