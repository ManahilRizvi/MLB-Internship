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
    #taking padding size from user
    sizePadding=int(input("Enter Padding Size: "))
    
    #padding can't be negative
    if sizePadding<0:
        print("Invalid size...")
    
    else:
        #calculating new size of image after padding
        #as padding is added on all 4 sides so 2* padding is added
        rowNew=rows+(2*sizePadding)
        colNew=columns+(2*sizePadding)
        paddedImg=np.zeros((rowNew, colNew, channel), dtype=np.uint8)
        #copying original image into center of new image as padding value shifts starting position
        for i in range(rows):
            for j in range(columns):
                for k in range(channel):
                    paddedImg[i+sizePadding][j+sizePadding][k]=img[i][j][k]
        
        paddedRgb=bgr_to_rgb(paddedImg)
        plt.imshow(paddedRgb)
        plt.title("Padded Image")
        plt.axis("off")
        plt.show()
