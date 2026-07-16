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
    #taking translation values from user where x is controlling left/right movement and y controls up/down movement
    transX=int(input("Enter translation in X direction: "))
    transY=int(input("Enter translation in Y direction: "))
    
    #values shouldn't move image completely outside
    if abs(transX)>=columns or abs(transY)>=rows:
        print("Invalid values...")
    
    else:
        transImg=np.zeros((rows, columns, channel), dtype=np.uint8)
        for i in range(rows):
            for j in range(columns):
                #calculating new position of current pixel
                rowNew=i+transY
                colNew=j+transX
                #copying pixel only if new position is inside image boundary
                if rowNew>=0 and rowNew<rows and colNew>=0 and colNew<columns:
                    for k in range(channel):
                        transImg[rowNew][colNew][k]=img[i][j][k]
        
        transRgb=bgr_to_rgb(transImg)
        plt.imshow(transRgb)
        plt.title("Translated Image")
        plt.axis("off")
        plt.show()
