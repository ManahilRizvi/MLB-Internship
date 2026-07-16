import cv2
import numpy as np
import matplotlib.pyplot as plt
#function to convert image from BGR to RGB formats
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
if img is None:
    print("No image....")
else:
    rows, columns, channel=img.shape
    print("Image Size: ", rows, "x", columns)
    #we used float32 as values contain decimals between 0 and 1
    normImg=np.zeros((rows, columns, channel), dtype=np.float32)
    
    #dividing each pixel value by 255 so that it falls between 0 and 1
    for i in range(rows):
        for j in range(columns):
            for k in range(channel):
                normImg[i][j][k]=img[i][j][k]/255.0
                
    #creating another image and converted it into uint8 so that original image can be same
    disImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    
    #converting pixels back to the range 0 to 255
    for i in range(rows):
        for j in range(columns):
            for k in range(channel):
                disImg[i][j][k]=int(normImg[i][j][k]*255)
    
    normRgb=bgr_to_rgb(disImg)
    #for comparison
    print("Img Pixels: ", img[0][0])
    print("Normalized Pixels: ", normImg[0][0])
    plt.imshow(normRgb)
    plt.title("Normalized Image")
    plt.axis("off")
    plt.show()
