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
    #new dimensions from user
    rowNew=int(input("Enter new row: "))
    colNew=int(input("Enter new column: "))
    
    #they should be greater than 0
    if rowNew<=0 or colNew<=0:
        print("Invalid points...")
    
    else:
        resizeImg=np.zeros((rowNew, colNew, channel), dtype=np.uint8)
        #calculating scaling ratio between original and new image
        scaleRow=rows/rowNew
        scaleCol=columns/colNew
        
        #resizing using nearest neighbor interpolation
        for i in range(rowNew):
            for j in range(colNew):
                #finding corresponding pixel in original image
                rowOld=int(i*scaleRow)
                colOld=int(j*scaleCol)
                
                #preventing index from going outside image boundary
                if rowOld>=rows:
                    rowOld=rows-1
                if colOld>=columns:
                    colOld=columns-1
                
                #copying pixel values from original image into resized image
                for k in range(channel):
                    resizeImg[i][j][k]=img[rowOld][colOld][k]
        
        resizeRgb=bgr_to_rgb(resizeImg)
        plt.imshow(resizeRgb)
        plt.title("Resized Image")
        plt.axis("off")
        plt.show()
