import cv2
import numpy as np
import matplotlib.pyplot as plt

def channelSep(image):
    rows, columns, channel=image.shape
    blue=np.zeros((rows, columns), dtype=np.uint8)
    green=np.zeros((rows, columns), dtype=np.uint8)
    red=np.zeros((rows, columns), dtype=np.uint8)

    for i in range(rows):
        for j in range(columns):
            blue[i][j]=image[i][j][0]
            green[i][j]=image[i][j][1]
            red[i][j]=image[i][j][2]
    return blue, green, red

def bgrToRgb(image):
    rows, columns, channel=image.shape
    rgbImg=np.zeros((rows, columns, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=image[i][j][0]
            green=image[i][j][1]
            red=image[i][j][2]

            rgbImg[i][j][0]=red
            rgbImg[i][j][1]=green
            rgbImg[i][j][2]=blue

    return rgbImg

def histogram(image):
    hisArr=np.zeros(256, dtype=int)
    rows, columns=image.shape
    for i in range(rows):
        for j in range(columns):
            pixel=image[i, j]
            hisArr[pixel]+=1
    return hisArr

img1=cv2.imread("dog.jpg")
img2=cv2.imread("jpeg_43-2.jpg")
if img1 is None or img2 is None:
    print("No image...")
    exit()

rgbImg1=bgrToRgb(img1)
rgbImg2=bgrToRgb(img2)
red1, green1, blue1=channelSep(rgbImg1)
red2, green2, blue2=channelSep(rgbImg2)

redHist1=histogram(red1)
greenHist1=histogram(green1)
blueHist1=histogram(blue1)

redHist2=histogram(red2)
greenHist2=histogram(green2)
blueHist2=histogram(blue2)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(rgbImg1)
plt.title("Image 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(rgbImg2)
plt.title("Image 2")
plt.axis("off")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(range(256), redHist1, color="red", label="red")
plt.plot(range(256), greenHist1, color="green", label="green")
plt.plot(range(256), blueHist1, color="blue", label="blue")
plt.title("RGB Histogram: Image 1")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(range(256), redHist2, color="red", label="red")
plt.plot(range(256), greenHist2, color="green", label="green")
plt.plot(range(256), blueHist2, color="blue", label="blue")
plt.title("RGB Histogram: Image 2")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#each channel of rgb has its own histogram
#red histogram shows distribution of red pixels
#green histogram shows distribution of green pixels
#blue histogram shows distribution of blue pixels
#different images produce different rgb histogram patterns
#dominant color channels has higher frequency values