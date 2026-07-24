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
            filterImg[i][j]=median       

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(noiseImg, cmap="gray")
    plt.title("Noisy Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(filterImg, cmap="gray")
    plt.title("Median Filter Image")
    plt.axis("off")
    plt.show()
#observations:
#it successfully removed most of salt and pepper noise from image
#edges were preserved better than mean and guassian filter
#image looked cleaner while maintaining important details
#it was more effective for impulse noise than averaging filters

#Which type of noise does it remove?
#it is designed to remove salt and pepper noise by replacing
#noisy pixels with median value of their neighboring pixels

#Why does it preserve edges better?
#it does not calculate an average instead it selects middle
#value after sorting neighboring pixels this prevents extreme noisy
#values from affecting result and helps preserve sharp edges and fine details