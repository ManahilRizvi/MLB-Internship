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

#an empty mask (black=0 and white=255)
mask=np.zeros((height, width), dtype=np.uint8)

#an empty segmented image
segmentedImg=np.zeros_like(rgb)

for i in range(height):
    for j in range(width):
            H=hsv[i, j, 0]
            S=hsv[i, j, 1]
            V=hsv[i, j, 2]
            
            #detecting red color
            if (H>=0 and H<=20) and (S>=0.4) and (V>=0.2):
                mask[i, j]=255#pixel is white
                segmentedImg[i, j]=rgb[i, j]#copying rgb pixel into segmented image
            else:
                mask[i, j]=0#pixel is black
                segmentedImg[i, j]=rgb[0, 0, 0]#background black

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(rgb)
plt.title("Image")
plt.axis("off")

#showing binary mask (white=detected object and black=background)
plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.axis("off")

#segmented image (only detected object is visible)
plt.subplot(1, 3, 3)
plt.imshow(segmentedImg)
plt.title("Segmented Image")
plt.axis("off")

plt.show()