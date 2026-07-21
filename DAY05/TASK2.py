import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("clear.jpeg")
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
        
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.plot(histogram)
    plt.title("Histogram")
    plt.xlabel("Pixel Intensity (0 to 255)")
    plt.ylabel("No of Pixels")
    
    plt.tight_layout()
    plt.show()

#histogram tells us how many pixels exists at each intensity level
#left side of histogram represents dark pixels
#right side represents bright pixels
#wide histogram means high contrast
#narrow histogram represents low contrast