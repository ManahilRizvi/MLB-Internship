import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image...")
else:
    rows, columns=img.shape
    noiseImg=img.copy()
    #adding salt and pepper noise
    for i in range(0, rows, 20):
        for j in range(0, columns, 20):
            noiseImg[i][j]=255#adding salt noise
            if i+10<rows and j+10<columns:
                noiseImg[i+10][j+10]=0
    filterImg=np.zeros((rows, columns), dtype=np.uint8)
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            #calculating sum of all 9 neighboring pixels
            total=(
                int(noiseImg[i-1][j-1])+
                int(noiseImg[i-1][j])+
                int(noiseImg[i-1][j+1])+
                int(noiseImg[i][j-1])+
                int(noiseImg[i][j])+
                int(noiseImg[i][j+1])+
                int(noiseImg[i+1][j-1])+
                int(noiseImg[i+1][j])+
                int(noiseImg[i+1][j+1])
            )
            #calculating average
            average=total//9
            filterImg[i][j]=average
    boxImg=np.zeros((rows, columns), dtype=np.uint8)
    #box filter kernel
    kernel=np.array([[1, 1, 1,], [1, 1, 1,], [1, 1, 1,]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(noiseImg[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            boxImg[i][j]=totalSum//9

    gaussianImg=np.zeros((rows, columns), dtype=np.uint8)
    #box filter kernel
    kernel=np.array([[1, 2, 1,], [2, 4, 2,], [1, 2, 1,]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(noiseImg[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing by dividing with sum of kernel values (16)
            gaussianImg[i][j]=totalSum//16

    medianImg=np.zeros((rows, columns), dtype=np.uint8)
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            pixels=[]#storing neighboring pixels
            #collecting all 9 neighboring pixels
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(noiseImg[i+m-1][j+n-1])#getting corresponding image pixel
                    pixels.append(pixel)
            #bubble sorting
            for x in range(len(pixels)-1):
                for y in range(len(pixels)-x-1):
                    if pixels[y]>pixels[y+1]:
                        temp=pixels[y]
                        pixels[y]=pixels[y+1]
                        pixels[y+1]=temp
            median=pixels[4]
            medianImg[i][j]=median

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(noiseImg, cmap="gray")
    plt.title("Noisy Image")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(filterImg, cmap="gray")
    plt.title("Mean Filter Image")
    plt.axis("off") 

    plt.subplot(2, 3, 4)
    plt.imshow(boxImg, cmap="gray")
    plt.title("Box Filter Image")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(gaussianImg, cmap="gray")
    plt.title("Gaussian Filter Image")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(medianImg, cmap="gray")
    plt.title("Median Filter Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

#comparison:
#MEAN FILTER
#noise removal ability=GOOD
#edge preservation=POOR
#blur level=HIGH
#best use case=RANDOM NOISE
#advantages=SIMPLE AND EASY TO IMPLEMENT
#disadvanatages=PRODUCES MORE BLUR AND WEAKENS EDGES

#BOX FILTER
#noise removal ability=GOOD
#edge preservation=POOR
#blur level=HIGH
#best use case=BASIC IMAGE SMOOTHING
#advantages=FAST AND SIMPLE
#disadvanatages=REMOVES EDGE DETAILS

#GAUSSIAN FILTER
#noise removal ability=VERY GOOD
#edge preservation=GOOD
#blur level=MEDIUM
#best use case=GAUSSIAN NOISE
#advantages=BETTER SMOOTHING WITH LESS EDGE LOSS
#disadvanatages=SLIGHTLY MORE COMPUTATIONALLY EXPENSIVE

#MEDIAN FILTER
#noise removal ability=EXCELLENT FOR SALT AND PEPPER NOISE
#edge preservation=EXCELLENT
#blur level=LOW
#best use case=SALT AND PEPPER NOISE
#advantages=PRESERVES EDGES WHILE REMOVING NOISE
#disadvanatages=SLOWER BECAUSE SORTING IS REQUIRED


#Which filter performs best for each scenario
#salt and pepper noise: median filter performs best because 
#it removes impulsive noise without averaging neighboring position

#gaussian noise: gaussian filter performs best because it 
#uses weighted averaging.

#image smoothing: mean filter and box filter are suitable because
#they are simple and computationally efficient

#edge preservation: median filter preserves edges best. gaussian
#filter also preserves edges better than mean and box filters.