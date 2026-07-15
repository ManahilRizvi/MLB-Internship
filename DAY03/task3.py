import cv2
import numpy as np
import matplotlib.pyplot as plt

#reading image OpenCv reads it in BGR format
img=cv2.imread("orange-flower.jpg")
rgb=img.copy()#copying image
height, width, channel=img.shape

#visiting every pixel and swapping blue and red channels
for i in range(height):
    for j in range(width):
        rgb[i, j, 0]=img[i, j, 2]
        rgb[i, j, 1]=img[i, j, 1]
        rgb[i, j, 2]=img[i, j, 0]

#making an empty hsv image it also has 3 channels
hsv=np.zeros((height, width, 3), dtype=float)

#visiting every pixel of RGB image
for i in range(height):
    for j in range(width):

        #reading values and converting them in range 0 to 1
        r=rgb[i, j, 0]/255.0
        g=rgb[i, j, 1]/255.0
        b=rgb[i, j, 2]/255.0

        #finding max and min value
        maxVal=max(r, g, b)
        minVal=min(r, g, b)
        difference=maxVal-minVal#find difference

        #calculating hue 
        if difference==0:
            H=0
        elif maxVal==r:
            H=(60*((g-b)/difference)+360)%360
        elif maxVal==g:
            H=(60*((b-r)/difference)+120)
        else:
            H=(60*((r-g)/difference)+240)
           
        #calculating saturation
        if maxVal==0:
            S=0
        else:
            S=difference/maxVal
        
        #calculating value
        V=maxVal

        #storing hsv values
        hsv[i, j, 0]=H
        hsv[i, j, 1]=S
        hsv[i, j, 2]=V

#making an empty hsl image it also has 3 channels
hsl=np.zeros((height, width, 3), dtype=float)
#visiting every pixel of RGB image
for i in range(height):
    for j in range(width):
        r=rgb[i, j, 0]/255.0
        g=rgb[i, j, 1]/255.0
        b=rgb[i, j, 2]/255.0

        maxVal=max(r, g, b)
        minVal=min(r, g, b)
        difference=maxVal-minVal

        #calculating hue
        if difference==0:
            H=0
        elif maxVal==r:
            H=(60*((g-b)/difference)+360)%360
        elif maxVal==g:
            H=(60*((b-r)/difference)+120)
        else:
            H=(60*((r-g)/difference)+240)

        #calculating lightness
        L=(maxVal+minVal)/2

        #calculating saturation
        if difference==0:
            S=0
        else:
            S=difference/(1-abs(2*L-1))
        
        #storing hsl values
        hsl[i, j, 0]=H
        hsl[i, j, 1]=S
        hsl[i, j, 2]=L

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

#an empty grayscale image
gray=np.zeros((height, width), dtype=np.uint8)
for i in range(height):
    for j in range(width):
        r=rgb[i, j, 0]
        g=rgb[i, j, 1]
        b=rgb[i, j, 2]

        #converting rgb into grayscale
        grayVal=(0.299*r)+(0.587*g)+(0.114*b)
        gray[i, j]=int(grayVal)

plt.figure(figsize=(18, 10))
plt.subplot(2, 3, 1)
plt.imshow(img[:, :, ::-1])#converting bgr to rgb for display
plt.title("Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(rgb)
plt.title("RGB")
plt.axis("off")

#hsv values normalized for visualization
plt.subplot(2, 3, 3)
hsvDis=np.zeros_like(rgb, dtype=float)
hsvDis[:, :, 0]=hsv[:, :, 0]/360
hsvDis[:, :, 1]=hsv[:, :, 1]
hsvDis[:, :, 2]=hsv[:, :, 2]
plt.imshow(hsvDis)
plt.title("HSV")
plt.axis("off")

#hsl values normalized for visualization
plt.subplot(2, 3, 4)
hslDis=np.zeros_like(rgb, dtype=float)
hslDis[:, :, 0]=hsl[:, :, 0]/360
hslDis[:, :, 1]=hsl[:, :, 1]
hslDis[:, :, 2]=hsl[:, :, 2]
plt.imshow(hslDis)
plt.title("HSL")
plt.axis("off")

#lab values normalized for visualization
plt.subplot(2, 3, 5)
labDis=np.zeros_like(rgb, dtype=float)
labDis[:, :, 0]=lab[:, :, 0]/100
labDis[:, :, 1]=(lab[:, :, 1]+128)/255
labDis[:, :, 2]=(lab[:, :, 2]+128)/255
plt.imshow(labDis)
plt.title("LAB")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(gray, cmap="gray")
plt.title("GRAYSCALE")
plt.axis("off")

plt.show()

#RGB used to display images on cameras, computers and mobile screens.

#HSV is used for color detection and object segmentation as 
#it separates color from brightness

#HSL is used in graphic design and image enhancement and many 
#computer applications because it separates brightness from
#color information

#Grayscale is used in edge detection, thresholding and other image processing.