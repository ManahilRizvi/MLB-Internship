import cv2
import numpy as np
import matplotlib.pyplot as plt
#reading image in grayscale mode
img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image")
else:
    rows, columns=img.shape#getting height and width
    print("Image Size: ", rows, "X", columns)
    #asking user for value
    padding=int(input("Enter size of padding: "))
    stride=int(input("Enter stride(1 or 2): "))

    #checking if it is valid
    if padding<0:
        print("Invalid size....")
    elif stride!=1 and stride!=2:
        print("Invalid..")
    else:
        #padding....
        #new image size after adding padding
        rowsP=rows+(2*padding)
        colsP=columns+(2*padding)
        paddedImg=np.zeros((rowsP, colsP), dtype=np.uint8)
        #copying original image into center of padded image
        for i in range(rows):
            for j in range(columns):
                paddedImg[i+padding][j+padding]=img[i][j]

        #kernel....
        blur=[[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        #calculating output image size according to stride
        newRow=((rowsP-3)//stride)+1
        newCol=((colsP-3)//stride)+1
        kernelImg=np.zeros((newRow, newCol), dtype=np.uint8)
        outputRow=0
        #convolution...
        #moving kernel over padded image using selected stride
        for i in range(0, rowsP-2, stride):
            outputCol=0
            for j in range(0, colsP-2, stride):
                #storing sum of neighboring pixels
                total=0
                #visiting every value of kernel
                for x in range(3):
                    for y in range(3):
                        pixel=int(paddedImg[i+x][j+y])
                        weight=blur[x][y]
                        total=total+(pixel*weight)
                total=int(total/9)
                if total<0:
                    total=0
                if total>255:
                    total=255
                kernelImg[outputRow][outputCol]=total
                outputCol+=1
            outputRow+=1

        print("Image: ", rows, "X", columns)
        print("Padded Image: ", rowsP, "X", colsP)
        print("Output Image: ", newRow, "X", newCol)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(paddedImg, cmap="gray")
        plt.title("Padded Image")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(kernelImg, cmap="gray")
        plt.title("Convolution Image")
        plt.axis("off")

        plt.show()
#step1: original grayscale image is loaded
#step2: manual zero padding is aroung image so that border
#pixels can also be processed
#step3: 3x3 blur kernel is applied manually using loops
#step4: kernel moves according to selected stride
#step5: final image is generated after convolution
