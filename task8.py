import cv2
import numpy as np
import matplotlib.pyplot as plt
#reading image in grayscale mode
img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image")
else:
    rows, columns=img.shape#getting height and width
    #creating a simple 3x3 blur kernel
    #every value is 1 because we want to take average of nearby pixels
    blur=[[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    #asking user for value
    stride=int(input("Enter stride(1 or 2): "))
    if stride!=1 and stride!=2:
        print("Invalid...")
    else:
        #calculating output image size based on stride
        rowNew=((rows-3)//stride)+1
        colNew=((columns-3)//stride)+1
        kernelImg=np.zeros((rowNew, colNew), dtype=np.uint8)
        outputRow=0
        #ignoring border pixels because 3x3 kernel can't fit there completely
        for i in range(0, rows-2, stride):
            outputCol=0
            for j in range(0, columns-2, stride):
                #to store sum of neighbor pixels
                total=0
                #moving through every value of 3x3 kernel
                for x in range(3):
                    for y in range(3):
                        #corresponding neighbor pixel
                        pixel=int(img[i+x][j+y])
                        #getting kernel value
                        weight=blur[x][y]
                        total=total+(pixel*weight)
                #dividing by 9 to calculate average pixel value
                total=int(total/9)
                #keeping value inside valid grayscale range
                if total<0:
                    total=0
                if total>255:
                    total=255
                kernelImg[outputRow][outputCol]=total
                outputCol+=1
            outputRow+=1

        print("Image Size: ", rows, "X", columns)
        print("Output Image Size: ", rowNew, "X", colNew)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(kernelImg, cmap="gray")
        plt.title("Stride= "+str(stride))
        plt.axis("off")

        plt.show()
#stride defines how many pixels kernel moves after each
#convolution operation
#stride=1; kernel moves one pixel at a time 
#more information is preserved and output image is larger
#stride=2; kernel moves two pixel at a time 
#fewer calculations are performed and output image is smaller
#larger stride reduces output size and may skip
#some image details during feature extraction