import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgbToGray(image):
    rows, columns, channel=image.shape
    gray=np.zeros((rows, columns),dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=image[i][j][0]
            green=image[i][j][1]
            red=image[i][j][2]
            grayVal=int(0.114*blue+0.587*green+0.299*red)
            gray[i][j]=grayVal
    return gray

def histogram(image):
    hisArr=np.zeros(256, dtype=int)
    rows, columns=image.shape
    for i in range(rows):
        for j in range(columns):
            pixel=image[i, j]
            hisArr[pixel]+=1
    return hisArr

img1=cv2.imread("cat.jpeg")
img2=cv2.imread("lowCon.jpg")
if img1 is None or img2 is None:
    print("No image...")
    exit()

imgGray1=rgbToGray(img1)
imgGray2=rgbToGray(img2)

img1Hist=histogram(imgGray1)
img2Hist=histogram(imgGray2)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(imgGray1, cmap="gray")
plt.title("Grayscale Image 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(imgGray2, cmap="gray")
plt.title("Grayscale Image 2")
plt.axis("off")
plt.show()


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(256), img1Hist)
plt.title("Histogram: Image 1")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.plot(range(256), img2Hist)
plt.title("Histogram: Image 2")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

#dark images have most histogram values on left side
#bright images have most values on right side
#low contrast images have narrow histogram
#high contrast images have wider histogram
