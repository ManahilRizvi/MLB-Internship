import cv2
import matplotlib.pyplot as plt
import numpy as np
#for grayscale image
img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image...")
else:
    rows, columns=img.shape
    #3x3 kernel
    gaussianImg3=img.copy()
    gaussKernel3=np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]])
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            total=0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    pixel=int(img[i+x][j+y])
                    weight=gaussKernel3[x+1][y+1]
                    total+=pixel*weight
            gaussianImg3[i][j]=total//16

    gaussianImg5=img.copy()
    gaussKernel5=np.array([[1, 4, 6, 4, 1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4, 6, 4, 1]])
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            total=0
            for x in range(-2, 3):
                for y in range(-2, 3):
                    pixel=int(img[i+x][j+y])
                    weight=gaussKernel5[x+2][y+2]
                    total+=pixel*weight
            gaussianImg5[i][j]=total//256

    gaussianImg7=img.copy()
    gaussKernel7=np.array([[0, 0, 1, 2, 1, 0, 0],
                            [0, 3, 13, 22, 13, 3, 0],
                            [1, 13, 59, 97, 59, 13, 1],
                            [2, 22, 97, 159, 97, 22, 2],
                            [1, 13, 59, 97, 59, 13, 1],
                            [0, 3, 13, 22, 13, 3, 0],
                            [0, 0, 1, 2, 1, 0, 0]])
    sum=np.sum(gaussKernel7)
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            total=0
            for x in range(-3, 4):
                for y in range(-3, 4):
                    pixel=int(img[i+x][j+y])
                    weight=gaussKernel7[x+3][y+3]
                    total+=pixel*weight
            gaussianImg7[i][j]=total//sum

    motionImg3=img.copy()
    motionKernel3=np.array([[0, 0, 0],
                           [1, 1, 1],
                           [0, 0, 0]])
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            total=0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    pixel=int(img[i+x][j+y])
                    weight=motionKernel3[x+1][y+1]
                    total+=pixel*weight
            motionImg3[i][j]=total//3

    motionImg5=img.copy()
    motionKernel5=np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            total=0
            for x in range(-2, 3):
                for y in range(-2, 3):
                    pixel=int(img[i+x][j+y])
                    weight=motionKernel5[x+2][y+2]
                    total+=pixel*weight
            motionImg5[i][j]=total//5

    motionImg7=img.copy()
    motionKernel7=np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]])
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            total=0
            for x in range(-3, 4):
                for y in range(-3, 4):
                    pixel=int(img[i+x][j+y])
                    weight=motionKernel7[x+3][y+3]
                    total+=pixel*weight
            motionImg7[i][j]=total//7

    medianImg3=img.copy()
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            values=[]
            for x in range(-1, 2):
                for y in range(-1, 2):
                    values.append(int(img[i+x][j+y]))
            for a in range(len(values)):
                for b in range(len(values)-1):
                    if values[b]>values[b+1]:
                        temp=values[b]
                        values[b]=values[b+1]
                        values[b+1]=temp
            medianImg3[i][j]=values[4]

    medianImg5=img.copy()
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            values=[]
            for x in range(-2, 3):
                for y in range(-2, 3):
                    values.append(int(img[i+x][j+y]))
            for a in range(len(values)):
                for b in range(len(values)-1):
                    if values[b]>values[b+1]:
                        temp=values[b]
                        values[b]=values[b+1]
                        values[b+1]=temp
            medianImg5[i][j]=values[12]

    medianImg7=img.copy()
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            values=[]
            for x in range(-3, 4):
                for y in range(-3, 4):
                    values.append(int(img[i+x][j+y]))
            for a in range(len(values)):
                for b in range(len(values)-1):
                    if values[b]>values[b+1]:
                        temp=values[b]
                        values[b]=values[b+1]
                        values[b+1]=temp
            medianImg7[i][j]=values[24]

    plt.figure(figsize=(20, 10))
    plt.subplot(3, 4, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(3, 4, 2)
    plt.imshow(gaussianImg3, cmap="gray")
    plt.title("Gaussian 3x3")
    plt.axis("off")

    plt.subplot(3, 4, 3)
    plt.imshow(gaussianImg5, cmap="gray")
    plt.title("Gaussian 5x5")
    plt.axis("off")

    plt.subplot(3, 4, 5)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(3, 4, 6)
    plt.imshow(motionImg3, cmap="gray")
    plt.title("Motion 3x3")
    plt.axis("off")

    plt.subplot(3, 4, 7)
    plt.imshow(motionImg5, cmap="gray")
    plt.title("Motion 5x5")
    plt.axis("off")

    plt.subplot(3, 4, 8)
    plt.imshow(motionImg7, cmap="gray")
    plt.title("Motion 7x7")
    plt.axis("off")

    plt.subplot(3, 4, 9)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(3, 4, 10)
    plt.imshow(medianImg3, cmap="gray")
    plt.title("Median 3x3")
    plt.axis("off")

    plt.subplot(3, 4, 11)
    plt.imshow(medianImg5, cmap="gray")
    plt.title("Median 5x5")
    plt.axis("off")

    plt.subplot(3, 4, 12)
    plt.imshow(medianImg7, cmap="gray")
    plt.title("Median 7x7")
    plt.axis("off")
    plt.tight_layout()
    plt.show()