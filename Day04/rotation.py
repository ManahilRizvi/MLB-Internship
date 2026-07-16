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
    #rotation angle from user
    angle=int(input("Enter angle of rotation(90, 180, 270): "))
    
    #only 90, 180 and 270 are allowed
    if angle!=90 and angle!=180 and angle!=270:
        print("Invalid angle...")
    
    else:
        #for 90 and 270, height and width are interchanged
        if angle==90 or angle==270:
            rotateImg=np.zeros((columns, rows, channel), dtype=np.uint8)
        #for 180 image size remains same
        else:
            rotateImg=np.zeros((rows, columns, channel), dtype=np.uint8)
        
        #if it is 90
        if angle==90:
            #copying each pixel to its new rotated position
            for i in range(rows):
                for j in range(columns):
                    for k in range(channel):
                        rotateImg[j][rows-1-i][k]=img[i][j][k]
        
        #if it is 180
        elif angle==180:
            #reversing both rows and columns
            for i in range(rows):
                for j in range(columns):
                    for k in range(channel):
                        rotateImg[rows-1-i][columns-1-j][k]=img[i][j][k]
        
        #if it is 270
        elif angle==270:
            #rotating image in opposite direction of 90
            for i in range(rows):
                for j in range(columns):
                    for k in range(channel):
                        rotateImg[columns-1-j][i][k]=img[i][j][k]
        
        rotateRgb=bgr_to_rgb(rotateImg)
        plt.imshow(rotateRgb)
        plt.title("Rotated Image")
        plt.axis("off")
        plt.show()
