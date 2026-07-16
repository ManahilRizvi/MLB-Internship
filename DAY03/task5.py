import cv2
import numpy as np
import matplotlib.pyplot as plt

#reading image OpenCv reads it in BGR format
img=cv2.imread("orange-flower.jpg")
rgb=img.copy()#copying image
height, width, channel=img.shape

#visiting every pixel and swapping blue and red values
for i in range(height):
    for j in range(width):
        rgb[i, j, 0]=img[i, j, 2]#red first
        rgb[i, j, 1]=img[i, j, 1]#green same
        rgb[i, j, 2]=img[i, j, 0]#blue last

#making an empty xyz image
xyz=np.zeros((height, width, 3), dtype=float)
for i in range(height):
    for j in range(width):
        r=rgb[i, j, 0]/255.0
        g=rgb[i, j, 1]/255.0
        b=rgb[i, j, 2]/255.0

        #applying gamma correction and this converts sRGB values into linear RGB values
        if r>0.04045:
            r=((r+0.055)/1.055)**2.4
        else:
            r=r/12.92

        if g>0.04045:
            g=((g+0.055)/1.055)**2.4
        else:
            g=g/12.92

        if b>0.04045:
            b=((b+0.055)/1.055)**2.4
        else:
            b=b/12.92
        
        #converting RGB to XYZ values
        X=(0.4124*r)+(0.3576*g)+(0.1805*b)
        Y=(0.2126*r)+(0.7152*g)+(0.0722*b)
        Z=(0.0193*r)+(0.1192*g)+(0.9505*b)

        #storing xyz values
        xyz[i, j, 0]=X
        xyz[i, j, 1]=Y
        xyz[i, j, 2]=Z

#reference white values
Xn=0.95047
Yn=1.00000
Zn=1.08883

#an empty lab image
lab=np.zeros((height, width, 3), dtype=float)
for i in range(height):
    for j in range(width):
        #normalize xyz values
        X=xyz[i, j, 0]/Xn
        Y=xyz[i, j, 1]/Yn
        Z=xyz[i, j, 2]/Zn
  
        #applying formula
        if X>0.008856:
            X=X**(1/3)
        else:
            X=(7.787*X)+(16/116)

        if Y>0.008856:
            Y=Y**(1/3)
        else:
            Y=(7.787*Y)+(16/116)
        
        if Z>0.008856:
            Z=Z**(1/3)
        else:
            Z=(7.787*Z)+(16/116)

        #calculating lab values
        L=(116*Y)-16
        A=500*(X-Y)
        B=200*(Y-Z)

        lab[i, j, 0]=L
        lab[i, j, 1]=A
        lab[i, j, 2]=B

#separate images for each LAB channels
channel_L=np.zeros((height, width), dtype=float)
channel_A=np.zeros((height, width), dtype=float)
channel_B=np.zeros((height, width), dtype=float)

for i in range(height):
    for j in range(width):
        channel_L[i, j]=lab[i, j, 0]#giving only lightness value to channel L
        channel_A[i, j]=lab[i, j, 1]#giving only green-red value to channel A
        channel_B[i, j]=lab[i, j, 2]#giving only blue-yellow value to channel B

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(rgb)
plt.title("RGB (original)")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(channel_L, cmap="gray")
plt.title("L CHANNEL")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(channel_A, cmap="gray")
plt.title("A CHANNEL")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(channel_B, cmap="gray")
plt.title("B CHANNEL")
plt.axis("off")

plt.show()

#L channel represents brightness of image.
#0 means black
#100 means white
#it doesn't have color information

#A channel represents color between green and red.
#negative value means green
#positive value means red

#B channel represents color between blue and yellow.
#negative value means blue
#positive value means yellow
