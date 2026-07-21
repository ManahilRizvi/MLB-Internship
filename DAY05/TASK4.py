import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("images.jpeg")
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
            
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    claheImg=clahe.apply(gray)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(equalized, cmap="gray")
    plt.title("Histogram Equalized Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(claheImg, cmap="gray")
    plt.title("Clahe Image")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

#histogram equalization enhances entire image at once
#it improves overall contrast but may increase noise
#CLAHE enhances small regions separately
#it prevents over enhancement and preserves more details
#clahe gives better result for dark, bright low contrast
