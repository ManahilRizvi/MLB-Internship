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
    choice=int(input("Enter choice(1 or 2): "))
    if choice!=1 and choice!=2:
        print("Invalid...")
    else:
        #for 3x3 kernel same padding is 1 pixel
        if choice==1:
            padding=int(input("Enter size of zero padding: "))
            title="Zero Padding"
        else:
            padding=1
            title="Same Padding"
        #value can't be negative
        if padding<0:
            print("Invalid size....")
        else:
            rowNew=rows+(2*padding)
            colNew=columns+(2*padding)
            paddedImg=np.zeros((rowNew, colNew), dtype=np.uint8)

            #copying image into center
            for i in range(rows):
                for j in range(columns):
                    paddedImg[i+padding][j+padding]=img[i][j]
            print("Image Size: ", rows, "X", columns)
            print("Padded Image Size: ", rowNew, "X", colNew)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap="gray")
            plt.title("Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(paddedImg, cmap="gray")
            plt.title(title)
            plt.axis("off")

            plt.show()
#padding adds extra pixels around image border
#in zero padding, added border pixels are filled with zero(black)
#in same padding, padding keeps output image size same after convolution
#when using a 3x3 kernel with stride=1
#padding helps kernel process border pixels and prevents loss of 
#image information at edges
