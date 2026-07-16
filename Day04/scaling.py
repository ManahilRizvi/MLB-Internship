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
    #scaling factor from user
    scaleFac=float(input("Enter Scaling Factor: "))
    
    #it must be greater than 0
    if scaleFac<=0:
        print("Invalid value...")
    
    else:
        #new dimensions of scaled image
        rowNew=int(rows*scaleFac)
        colNew=int(columns*scaleFac)
        scaleImg=np.zeros((rowNew, colNew, channel), dtype=np.uint8)
        for i in range(rowNew):
            for j in range(colNew):
                #finding corresponding pixel in original image then map new image back to original image
                rowOld=int(i/scaleFac)
                colOld=int(j/scaleFac)
                
                #preventing index from going outside image boundary
                if rowOld>=rows:
                    rowOld=rows-1
                if colOld>=columns:
                    colOld=columns-1 
                #copying pixel values
                for k in range(channel):
                    scaleImg[i][j][k]=img[rowOld][colOld][k]
        
        scaleRgb=bgr_to_rgb(scaleImg)
        plt.imshow(scaleRgb)
        plt.title("Scaled Image")
        plt.axis("off")
        plt.show()
