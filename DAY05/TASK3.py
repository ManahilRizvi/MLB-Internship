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
            
    histogram=np.zeros(256, dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            pixel=gray[i][j]
            histogram[pixel]+=1
            
    cdf=np.zeros(256, dtype=np.uint8)
    cdf[0]=histogram[0]
    for i in range(1, 256):
        cdf[i]=cdf[i-1]+histogram[i]
        
    totalPix=rows*columns
    normCdf=np.zeros(256, dtype=np.uint8)
    for i in range(256):
        n=(cdf[i]*255)/totalPix
        normCdf[i]=int(n)
        
    equalized=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            pixel=gray[i][j]
            equalized[i][j]=normCdf[pixel]
            
    equHist=np.zeros(256, dtype=int)
    for i in range(rows):
        for j in range(columns):
            pixel=equalized[i][j]
            equHist[pixel]+=1
        
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Image")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.plot(histogram)
    plt.title("Histogram")
    
    plt.subplot(2, 2, 3)
    plt.imshow(equalized, cmap="gray")
    plt.title("Equalized Image")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.plot(equHist)
    plt.title("Equalized Histogram")
    
    plt.tight_layout()
    plt.show()