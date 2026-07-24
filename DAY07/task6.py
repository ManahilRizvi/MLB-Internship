import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image...")
else:
    rows, columns=img.shape
    filterImg3=np.zeros((rows, columns), dtype=np.uint8)
    filterImg5=np.zeros((rows, columns), dtype=np.uint8)
    filterImg7=np.zeros((rows, columns), dtype=np.uint8)
    #for 3x3 kernel
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            #calculating sum of all 9 neighboring pixels
            total=0
            for m in range(3):
                for n in range(3):
                    total=total+int(img[i+m-1][j+n-1])
            #calculating average
            average=total//9
            filterImg3[i][j]=average
            
    #for 5x5 kernel
    #loop through all pixels except border pixels
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            #calculating sum of all 9 neighboring pixels
            total=0
            for m in range(5):
                for n in range(5):
                    total=total+int(img[i+m-2][j+n-2])
            #calculating average
            average=total//25
            filterImg5[i][j]=average
            
    #for 7x7 kernel
    #loop through all pixels except border pixels
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            #calculating sum of all 9 neighboring pixels
            total=0
            for m in range(7):
                for n in range(7):
                    total=total+int(img[i+m-3][j+n-3])
            #calculating average
            average=total//49
            filterImg7[i][j]=average
    
    #box filter
    boxImg3=np.zeros((rows, columns), dtype=np.uint8)
    boxImg5=np.zeros((rows, columns), dtype=np.uint8)
    boxImg7=np.zeros((rows, columns), dtype=np.uint8)
    
    #for 3x3 kernel
    kernel3=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel3[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            boxImg3[i][j]=totalSum//9
            
    #for 5x5 kernel
    kernel5=np.array([[1, 1, 1, 1, 1], 
                      [1, 1, 1, 1, 1], 
                      [1, 1, 1, 1, 1], 
                      [1, 1, 1, 1, 1], 
                      [1, 1, 1, 1, 1]])
    #loop through all pixels except border pixels
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            totalSum=0
            for m in range(5):#through rows
                for n in range(5):#through columns
                    pixel=int(img[i+m-2][j+n-2])#getting corresponding image pixel
                    weight=kernel5[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            boxImg5[i][j]=totalSum//25
            
    #for 7x7 kernel
    kernel7=np.array([[1, 1, 1, 1, 1, 1, 1], 
                      [1, 1, 1, 1, 1, 1, 1], 
                      [1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1]])
    #loop through all pixels except border pixels
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            totalSum=0
            for m in range(7):#through rows
                for n in range(7):#through columns
                    pixel=int(img[i+m-3][j+n-3])#getting corresponding image pixel
                    weight=kernel7[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            boxImg7[i][j]=totalSum//49
    
    #gaussian filter
    gaussianImg3=np.zeros((rows, columns), dtype=np.uint8)
    gaussianImg5=np.zeros((rows, columns), dtype=np.uint8)
    gaussianImg7=np.zeros((rows, columns), dtype=np.uint8)
    
    #for 3x3 kernel
    kernel3=np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel3[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            gaussianImg3[i][j]=totalSum//16
            
    #for 5x5 kernel
    kernel5=np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
    #loop through all pixels except border pixels
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            totalSum=0
            for m in range(5):#through rows
                for n in range(5):#through columns
                    pixel=int(img[i+m-2][j+n-2])#getting corresponding image pixel
                    weight=kernel5[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            gaussianImg5[i][j]=totalSum//256
            
    #for 7x7 kernel
    kernel7=np.array([[0, 0, 1, 2, 1, 0, 0], 
                      [0, 3, 13, 22, 13, 3, 0], 
                      [1, 13, 59, 97, 59, 13, 1],
                      [2, 22, 97, 159, 97, 22, 2],
                      [1, 13, 59, 97, 59, 13, 1],
                      [0, 3, 13, 22, 13, 3, 0],
                      [0, 0, 1, 2, 1, 0, 0]])
    #loop through all pixels except border pixels
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            totalSum=0
            for m in range(7):#through rows
                for n in range(7):#through columns
                    pixel=int(img[i+m-3][j+n-3])#getting corresponding image pixel
                    weight=kernel7[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            gaussianImg7[i][j]=totalSum//1003
    
    #median filter
    medianImg3=np.zeros((rows, columns), dtype=np.uint8)
    medianImg5=np.zeros((rows, columns), dtype=np.uint8)
    medianImg7=np.zeros((rows, columns), dtype=np.uint8)
    
    #for 3x3 
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            pixels=[]#storing neighboring pixels
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    pixels.append(pixel)
            #bubble sorting
            for x in range(len(pixels)-1):
                for y in range(len(pixels)-x-1):
                    if pixels[y]>pixels[y+1]:
                        temp=pixels[y]
                        pixels[y]=pixels[y+1]
                        pixels[y+1]=temp
            #normalizing
            medianImg3[i][j]=pixels[4]
            
    #for 5x5 kernel
    #loop through all pixels except border pixels
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            pixels=[]#storing neighboring pixels
            for m in range(5):#through rows
                for n in range(5):#through columns
                    pixel=int(img[i+m-2][j+n-2])#getting corresponding image pixel
                    pixels.append(pixel)
            #bubble sorting
            for x in range(len(pixels)-1):
                for y in range(len(pixels)-x-1):
                    if pixels[y]>pixels[y+1]:
                        temp=pixels[y]
                        pixels[y]=pixels[y+1]
                        pixels[y+1]=temp
            medianImg5[i][j]=pixels[12]
            
    #for 7x7 kernel
    #loop through all pixels except border pixels
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            pixels=[]#storing neighboring pixels
            for m in range(7):#through rows
                for n in range(7):#through columns
                    pixel=int(img[i+m-3][j+n-3])#getting corresponding image pixel
                    pixels.append(pixel)
            #bubble sorting
            for x in range(len(pixels)-1):
                for y in range(len(pixels)-x-1):
                    if pixels[y]>pixels[y+1]:
                        temp=pixels[y]
                        pixels[y]=pixels[y+1]
                        pixels[y+1]=temp
            medianImg7[i][j]=pixels[24]
            
    plt.figure(figsize=(16, 16))
    plt.subplot(4, 4, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(4, 4, 2)
    plt.imshow(filterImg3, cmap="gray")
    plt.title("3x3 Mean Filter")
    plt.axis("off")
    
    plt.subplot(4, 4, 3)
    plt.imshow(filterImg5, cmap="gray")
    plt.title("5x5 Mean Filter")
    plt.axis("off")
    
    plt.subplot(4, 4, 4)
    plt.imshow(filterImg7, cmap="gray")
    plt.title("7x7 Mean Filter")
    plt.axis("off")

    plt.subplot(4, 4, 5)
    plt.imshow(boxImg3, cmap="gray")
    plt.title("3x3 Box Filter")
    plt.axis("off")

    plt.subplot(4, 4, 6)
    plt.imshow(boxImg5, cmap="gray")
    plt.title("5x5 Box Filter")
    plt.axis("off")

    plt.subplot(4, 4, 7)
    plt.imshow(boxImg7, cmap="gray")
    plt.title("7x7 Box Filter")
    plt.axis("off")
    plt.subplot(4, 4, 8)
    plt.axis("off")

    plt.subplot(4, 4, 9)
    plt.imshow(gaussianImg3, cmap="gray")
    plt.title("3x3 Gaussian Filter")
    plt.axis("off")

    plt.subplot(4, 4, 10)
    plt.imshow(gaussianImg5, cmap="gray")
    plt.title("5x5 Gaussian Filter")
    plt.axis("off")

    plt.subplot(4, 4, 11)
    plt.imshow(gaussianImg7, cmap="gray")
    plt.title("7x7 Gaussian Filter")
    plt.axis("off")
    plt.subplot(4, 4, 12)
    plt.axis("off")

    plt.subplot(4, 4, 13)
    plt.imshow(medianImg3, cmap="gray")
    plt.title("3x3 Median Filter")
    plt.axis("off")

    plt.subplot(4, 4, 14)
    plt.imshow(medianImg5, cmap="gray")
    plt.title("5x5 Median Filter")
    plt.axis("off")

    plt.subplot(4, 4, 15)
    plt.imshow(medianImg7, cmap="gray")
    plt.title("7x7 Median Filter")
    plt.axis("off")
    plt.subplot(4, 4, 16)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    
#observations:
#3x3 kernel
#mean filter: removes a small amount of noise while keeping most image details.
#box filter: removes a small amount of noise while keeping most image details.
#gaussian filter: removes a small amount of noise while preserving edges better than mean and box filter
#median filter: removes a small amount of salt and pepper noise while preserving image edges

#5x5 kernel
#mean filter: removes more noise than 3x3 filter but slightly blurs edges.
#box filter: removes more noise but slightly blurs edges.
#gaussian filter: removes more noise and produces a smoother image with moderate blur
#median filter: removes more noise than 3x3 but starts smoothing fine details

#7x7 kernel
#mean filter: removes maximum amount of noise but produces highest blur and reduces image sharpness.
#box filter: removes maximum amount of noise but produces more blur and reduces image sharpness.
#gaussian filter: removes maximum noise but increases blur and reduces image sharpness
#median filter: removes maximum noise but may remove small image details

#comparisons:
#for mean filter;
#image sharpness= 3x3>5x5>7x7
#noise reduction= 7x7>5x5>3x3
#blur level= 7x7>5x5>3x3
#processing time= 3x3>5x5>7x7

#for box filter;
#image sharpness= 3x3>5x5>7x7
#noise reduction= 7x7>5x5>3x3
#blur level= 7x7>5x5>3x3
#processing time= 3x3>5x5>7x7

#for gaussian filter;
#image sharpness= 3x3>5x5>7x7
#noise reduction= 7x7>5x5>3x3
#blur level= 7x7>5x5>3x3
#processing time= 3x3>5x5>7x7

#for median filter;
#image sharpness= 3x3>5x5>7x7
#noise reduction= 7x7>5x5>3x3
#blur level= 7x7>5x5>3x3
#processing time= 3x3>5x5>7x7