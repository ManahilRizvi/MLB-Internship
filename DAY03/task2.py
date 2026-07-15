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

#make three empty images to store red, green and blue channel
redImg=np.zeros_like(rgb)
greenImg=np.zeros_like(rgb)
blueImg=np.zeros_like(rgb)

for i in range(height):
    for j in range(width):
        redImg[i, j, 0]=rgb[i, j, 0]#only red values
        greenImg[i, j, 1]=rgb[i, j, 1]#only green values
        blueImg[i, j, 2]=rgb[i, j, 2]#only blue values

#making an empty image to store merging of these channels
finalImg=np.zeros_like(rgb)
for i in range(height):
    for j in range(width):
        finalImg[i, j, 0]=redImg[i, j, 0]
        finalImg[i, j, 1]=greenImg[i, j, 1]
        finalImg[i, j, 2]=blueImg[i, j, 2]

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(rgb)
plt.title("RGB (original)")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(redImg)
plt.title("RED CHANNEL")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(greenImg)
plt.title("GREEN CHANNEL")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(blueImg)
plt.title("BLUE CHANNEL")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(finalImg)
plt.title("Merged Image")
plt.axis("off")

plt.show()