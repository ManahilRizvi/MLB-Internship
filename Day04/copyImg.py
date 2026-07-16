import cv2
import numpy as np
import matplotlib.pyplot as plt
#function to convert image from BGR to RGB format
def bgr_to_rgb(img):
    rows, columns, channel=img.shape
    rgb=np.zeros((rows, columns, channel), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            rgb[i][j][0]=img[i][j][2]#red color
            rgb[i][j][1]=img[i][j][1]#green color
            rgb[i][j][2]=img[i][j][0]#blue color
    return rgb

img=cv2.imread("jpeg_43-2.jpg")
if img is None:#checking if there is an image or not
    print("No image....")
else:
    rows, columns, channel=img.shape
    #an empty image which will store the copied pixels 
    copyImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    #copying every pixel using loops
    for i in range(rows):
        for j in range(columns):
            for k in range(channel):
                copyImg[i][j][k]=img[i][j][k]
    
    #converting both images into RGB format
    origRgb=bgr_to_rgb(img)
    copyRgb=bgr_to_rgb(copyImg)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(origRgb)
    plt.title("Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(copyRgb)
    plt.title("Copy Image")
    plt.axis("off")
    
    plt.show()
