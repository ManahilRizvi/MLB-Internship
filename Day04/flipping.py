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
    #asking user which type of flip they want
    print("\nChoose Flip Type...")
    print("1. Horizontal Flip")
    print("2. Vertical Flip")
    print("3. Horizontal+Vertical Flip")
    
    ans=int(input("Enter answer(1-3): "))
    
    if ans<1 or ans>3:
        print("Invalid answer...")
    
    else:
        flipImg=np.zeros((rows, columns, channel), dtype=np.uint8)
        
        #for horizontal flip
        if ans==1:
            #reversing column positions while keeping rows same
            for i in range(rows):
                for j in range(columns):
                    for k in range(channel):
                        flipImg[i][columns-1-j][k]=img[i][j][k]
        
        #for vertical flip
        elif ans==2:
            #reversing row positions while keeping columns same 
            for i in range(rows):
                for j in range(columns):
                    for k in range(channel):
                        flipImg[rows-1-i][j][k]=img[i][j][k]
        
        #for horizontal+vertical flip
        elif ans==3:
            #reversing both rows and columns at same time
            for i in range(rows):
                for j in range(columns):
                    for k in range(channel):
                        flipImg[rows-1-i][columns-1-j][k]=img[i][j][k]
        
        flipRgb=bgr_to_rgb(flipImg)
        plt.imshow(flipRgb)
        plt.title("Flipped Image")
        plt.axis("off")
        plt.show()
