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

def blendImg(img1, img2):
    rows, columns, channel=img1.shape
    stitch=np.zeros((rows, columns, channel), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            pix1=img1[i][j]
            pix2=img2[i][j]
            exists1=False
            for c in range(channel):
                if pix1[c]!=0:
                    exists1=True
            exists2=False
            for c in range(channel):
                if pix2[c]!=0:
                    exists2=True
            if exists1 and not exists2:
                for c in range(channel):
                    stitch[i][j][c]=pix1[c]
            elif not exists1 and exists2:
                for c in range(channel):
                    stitch[i][j][c]=pix2[c]
            elif exists1 and exists2:
                for c in range(channel):
                    value=(0.5*float(pix1[c])+0.5*float(pix2[c]))
                    stitch[i][j][c]=int(value)
            else:
                for c in range(channel):
                    stitch[i][j][c]=0
    return stitch

img1=cv2.imread("book1.jpeg")
alignImg=cv2.imread("alignImg.png")
if img1 is None or alignImg is None:
    print("No image...")
    exit()

rows=img1.shape[0]
columns=img1.shape[1]
alignImg=cv2.resize(alignImg, (columns, rows))
stitchImg=blendImg(img1, alignImg)
img1Rgb=bgrToRgb(img1)
alignRgb=bgrToRgb(alignImg)
stitchRgb=bgrToRgb(stitchImg)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img1Rgb)
plt.title("Reference Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(alignRgb)
plt.title("Aligned Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(stitchRgb)
plt.title("Stitched and Blended")
plt.axis("off")
plt.tight_layout()
plt.show()