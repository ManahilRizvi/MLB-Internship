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
    
    #asking user for crop coordinates 
    rowStart=int(input("Enter starting row: "))
    rowEnd=int(input("Enter ending row: "))
    colStart=int(input("Enter starting column: "))
    colEnd=int(input("Enter ending column: "))
    
    #checking if user input is inside the image
    if (rowStart<0 or rowEnd>rows or colStart<0 or colEnd>columns or rowStart>=rowEnd or colStart>=colEnd):
        print("Invalid points...")
    
    else:
        #calculating height and width of cropped image
        rowsCrop=rowEnd-rowStart
        colsCrop=colEnd-colStart
        cropImg=np.zeros((rowsCrop, colsCrop, channel), dtype=np.uint8)
        
        #copying only selected area pixel by pixel
        for i in range(rowsCrop):
            for j in range(colsCrop):
                for k in range(channel):
                    cropImg[i][j][k]=img[rowStart+i][colStart+j][k]
        
        cropRgb=bgr_to_rgb(cropImg)
        plt.imshow(cropRgb)
        plt.title("Cropped Image")
        plt.axis("off")
        plt.show()
