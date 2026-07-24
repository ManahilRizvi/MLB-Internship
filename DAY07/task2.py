import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image...")
else:
    rows, columns=img.shape
    filterImg=np.zeros((rows, columns), dtype=np.uint8)
    #box filter kernel
    kernel=np.array([[1, 1, 1,], [1, 1, 1,], [1, 1, 1,]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            filterImg[i][j]=totalSum//9

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(filterImg, cmap="gray")
    plt.title("Box Filter Image")
    plt.axis("off")
    plt.show()
#observations:
#it successfully reduces random noise in image
#filtered image appeared smoother than original image
#some image details and edges became blurred due to averaging
#output of normalized box filter was almost identical to mean filter
#filter worked well for smoothing but was not suitable for preserving sharp edges