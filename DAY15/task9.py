import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("dog.jpg")
height=img.shape[0]
width=img.shape[1]
rgb=np.zeros_like(img)
for i in range(height):
    for j in range(width):
        blue=img[i][j][0]
        green=img[i][j][1]
        red=img[i][j][2]

        rgb[i][j][0]=red
        rgb[i][j][1]=green
        rgb[i][j][2]=blue
#flipping
flip=np.zeros_like(rgb)
for i in range(height):
    for j in range(width):
        flip[i][j]=rgb[i][width-1-j]
#rotation
angle=30
radian=np.deg2rad(angle)
cosVal=np.cos(radian)
sinVal=np.sin(radian)
centerX=width//2
centerY=height//2
rotate=np.zeros_like(rgb)
for i in range(height):
    for j in range(width):
        x=j-centerX
        y=i-centerY
        newX=int(x*cosVal-y*sinVal)
        newY=int(x*sinVal+y*cosVal)
        newX=newX+centerX
        newY=newY+centerY
        if newX>=0 and newX<width and newY>=0 and newY<height:
            rotate[newY][newX]=rgb[i][j]
#cropping
cropHeight=int(height*0.8)
cropWidth=int(width*0.8)
startY=(height-cropHeight)//2
startX=(width-cropWidth)//2
crop=np.zeros((cropHeight, cropWidth, 3), dtype=np.uint8)
for i in range(cropHeight):
    for j in range(cropWidth):
        crop[i][j]=rgb[startY+i][startX+j]
#scaling
scale=0.7
newH=int(height*scale)
newW=int(width*scale)
scaleImg=np.zeros((newH, newW, 3), dtype=np.uint8)
for i in range(newH):
    for j in range(newW):
        oldI=int(i/scale)
        oldJ=int(j/scale)
        if oldI<height and oldJ<width:
            scaleImg[i][j]=rgb[oldI][oldJ]
#brightness
brightVal=50
brightImg=np.zeros_like(rgb)
for i in range(height):
    for j in range(width):
        for k in range(3):
            value=int(rgb[i][j][k])+brightVal
            if value>255:
                value=255
            if value<0:
                value=0
            brightImg[i][j][k]=value
#contrast
contrastFact=1.5
contrast=np.zeros_like(rgb)
for i in range(height):
    for j in range(width):
        for k in range(3):
            value=int(rgb[i][j][k]*contrastFact)
            if value>255:
                value=255
            if value<0:
                value=0
            contrast[i][j][k]=value

plt.figure(figsize=(12, 8))
plt.subplot(2, 4, 1)
plt.imshow(rgb)
plt.title("Image")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(flip)
plt.title("Horizontal Flip")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(rotate)
plt.title("Rotation")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(crop)
plt.title("Crop")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(scaleImg)
plt.title("Scaling")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(brightImg)
plt.title("Brightness")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(contrast)
plt.title("Contrast")
plt.axis("off")
plt.tight_layout()
plt.show()