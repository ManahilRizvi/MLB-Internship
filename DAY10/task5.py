import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalization(image):
    image=np.abs(image)
    minVal=np.min(image)
    maxVal=np.max(image)
    result=np.zeros(image.shape, dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            if maxVal!=minVal:
                value=((image[i][j]-minVal)/(maxVal-minVal))*255
                result[i][j]=int(value)
    return result

#it scales values using only maximum value
#so that weak edges become visible
def DoGNorm(image):
    image=np.abs(image)
    maxVal=np.max(image)
    result=np.zeros(image.shape, dtype=np.uint8)
    if maxVal==0:
        return result
    for i in range(rows):
        for j in range(columns):
            value=(image[i][j]/maxVal)*255
            if value>255:
                value=255
            result[i][j]=int(value)
    return result      

img=cv2.imread("dog.jpg")
if img is None:
    print("No image...")
else:
    rows, columns, channel=img.shape
    grayImg=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=img[i][j][0]
            green=img[i][j][1]
            red=img[i][j][2]
            grayVal=int(0.114*blue+0.587*green+0.299*red)

            if grayVal>255:
                grayVal=255
            elif grayVal<0:
                grayVal=0
            grayImg[i][j]=grayVal

    kernel=[[1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]]

    gaussImg1=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(grayImg[i+m][j+n])
                    value+=pixel*kernel[m+1][n+1]
            gaussImg1[i][j]=value/16

#secomd gaussian filter on already blurred image this
#produces stronger smoothing
    gaussImg2=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(gaussImg1[i+m][j+n])
                    value+=pixel*kernel[m+1][n+1]
            gaussImg2[i][j]=value/16

#subtracting two blurred images 
#differenec shows edges and fine image details
    dogImg=gaussImg1-gaussImg2
    dogDisplay=DoGNorm(dogImg)
    gaussDisplay1=gaussImg1.astype(np.uint8)
    gaussDisplay2=gaussImg2.astype(np.uint8)

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(grayImg, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gaussDisplay1, cmap="gray")
    plt.title("Gaussian Blur 1")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(gaussDisplay2, cmap="gray")
    plt.title("Gaussian Blur 2")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(dogDisplay, cmap="gray")
    plt.title("DoG Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

#Difference of Gaussian is obtained by subtracting
#two gaussian blurred images with different smoothing levels
#first gaussian blur preserves more details while second
#blur smooths image further. their difference highlights edges and 
#fine details. compared to LoG, DoG is faster computationally
#and provides good approximation of laplacian of gaussian.