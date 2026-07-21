import cv2
import numpy as np
import matplotlib.pyplot as plt
#reading image in grayscale mode
img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image")
else:
    rows, columns=img.shape#getting height and width
    
    minPix=img[0][0]
    maxPix=img[0][0]
    totalInt=0#to calculate average intensity
    range1=0#for range 0 to 50
    range2=0#for range 51 to 100
    range3=0#for range 101 to 150
    range4=0#for range 151 to 200
    range5=0#for range 201 to 255
    histogram=np.zeros(256, dtype=int)
    #loop through every pixel 
    for i in range(rows):
        for j in range(columns):
            pixel=img[i][j]
            if pixel<minPix:
                minPix=pixel
            if pixel>maxPix:
                maxPix=pixel
            #calculating total intensity
            totalInt+=pixel
            #frequency of histogram
            histogram[pixel]+=1

            #pixels in different intensity ranges
            if pixel<=50:
                range1+=1
            elif pixel<=100:
                range2+=1
            elif pixel<=150:
                range3+=1
            elif pixel<=200:
                range4+=1
            else:
                range5+=1

    #calculating average intensity        
    average=totalInt/(rows*columns)
    print("Maximum pixel intensity: ", maxPix)   
    print("Minimum pixel intensity: ", minPix)
    print("Average pixel Intensity: ", average)
    print("Pixel in these Intensity Range...")
    print("0 to 50: ", range1)
    print("51 to 100: ", range2)
    print("101 to 150: ", range3)
    print("151 to 200: ", range4)
    print("201 to 255: ", range5)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.plot(histogram)
    plt.title("Intensity Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("No of pixels")

    plt.tight_layout()
    plt.show()