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
    print("No image....")
    exit()

srcPoints=[[100, 100],
           [500, 100],
           [500, 500],
           [100, 500]]
destPoints=[[80, 120],
            [520, 80],
            [550, 520],
            [70, 500]]

H=dlt(srcPoints, destPoints)
print("Homography Matrix: ")
print(H)
outRows=img2.shape[0]
outCols=img2.shape[1]
result=warp(img1, H, outRows, outCols)
opencvResult=cv2.warpPerspective(img1, H, (outCols, outRows))
img1Rgb=bgrToRgb(img1)
img2Rgb=bgrToRgb(img2)
rgb=bgrToRgb(result)
opencvRgb=bgrToRgb(opencvResult)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(img1Rgb)
plt.title("Source Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(img2Rgb)
plt.title("Destination Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(rgb)
plt.title("Warping")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(opencvRgb)
plt.title("OpenCV Warp Perspective")
plt.axis("off")
plt.tight_layout()
plt.show()