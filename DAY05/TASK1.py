import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("orange-flower.jpg")
if img is None:
    print("No image")
else:
    rows, columns, channel=img.shape
    gray=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=img[i][j][0]
            green=img[i][j][1]
            red=img[i][j][2]
            intensity=int(0.299*red+0.587*green+0.114*blue)
            gray[i][j]=intensity
    threshold=int(input("Enter value of threshold (0 to 255): "))
    
    binaryThreshold=np.zeros((rows, columns), dtype=np.uint8)
    binaryInverse=np.zeros((rows, columns), dtype=np.uint8)
    truncateThreshold=np.zeros((rows, columns), dtype=np.uint8)
    zero=np.zeros((rows, columns), dtype=np.uint8)
    invToZero=np.zeros((rows, columns), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(columns):
            pixel=gray[i][j]
            #binary
            if pixel>threshold:
                binaryThreshold[i][j]=255
            else:
                binaryThreshold[i][j]=0
                
            #binaryInverse
            if pixel>threshold:
                binaryInverse[i][j]=0
            else:
                binaryInverse[i][j]=255
            
            #truncate
            if pixel>threshold:
                truncateThreshold[i][j]=threshold
            else:
                truncateThreshold[i][j]=pixel
            
            #to zero threshold
            if pixel>threshold:
                zero[i][j]=pixel
            else:
                zero[i][j]=0
                
            #inverse to zero
            if pixel>threshold:
                invToZero[i][j]=0
            else:
                invToZero[i][j]=pixel
        
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Image")
    plt.axis("off")
    
    plt.subplot(2, 3, 2)
    plt.imshow(binaryThreshold, cmap="gray")
    plt.title("Binary Threshold")
    plt.axis("off")
    
    plt.subplot(2, 3, 3)
    plt.imshow(binaryInverse, cmap="gray")
    plt.title("Binary Inverse Threshold")
    plt.axis("off")
    
    plt.subplot(2, 3, 4)
    plt.imshow(truncateThreshold, cmap="gray")
    plt.title("Truncate Threshold")
    plt.axis("off")
    
    plt.subplot(2, 3, 5)
    plt.imshow(zero, cmap="gray")
    plt.title("To Zero Threshold")
    plt.axis("off")
    
    plt.subplot(2, 3, 6)
    plt.imshow(invToZero, cmap="gray")
    plt.title("Inverse To Zero Threshold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()