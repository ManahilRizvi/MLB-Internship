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
    identity=[[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    blur=[[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    sharp=[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    edge=[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

    print("Which Kernel?")
    print("1. Identity Kernel")
    print("2. Blur Kernel")
    print("3. Sharpen Kernel")
    print("4. Edge Detection Kernel")
    choice=int(input("Enter choice(1 to 4): "))
    if choice==1:
        #identity kernel
        kernel=identity
        title="Identity Kernel"
    elif choice==2:
        #blur kernel
        kernel=blur
        title="Blur Kernel"
    elif choice==3:
        #sharp kernel
        kernel=sharp
        title="Sharpen Kernel"
    elif choice==4:
        #edge detection kernel
        kernel=edge
        title="Edge Detection Kernel"
    else:
        print("Invalid...")
        exit()
    kernelImg=np.zeros((rows, columns), dtype=np.uint8)
    #ignoring border pixels because 3x3 kernel can't fit there completely
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            #to store sum of neighbor pixels
            total=0
            #moving through every value of 3x3 kernel
            for x in range(3):
                for y in range(3):
                    #corresponding neighbor pixel
                    pixel=int(img[i+x-1][j+y-1])
                    #getting kernel value
                    weight=kernel[x][y]
                    total=total+(pixel*weight)
            if choice==2:
                #dividing by 9 to calculate average pixel value
                total=int(total/9)
            #keeping value inside valid grayscale range
            if total<0:
                total=0
            if total>255:
                total=255
            kernelImg[i][j]=total

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(kernelImg, cmap="gray")
    plt.title(title)
    plt.axis("off")

    plt.show()

#identity kernel returns almost same image because only center
#pixel is used
#blur kernel averages neighboring pixels to reduce noise
#and smooth image
#sharpen kernel increases edge details and makes image appear clear
#edge detection kernel highlights object boundaries by detecting 
#sudden intensity changes
