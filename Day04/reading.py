import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("jpeg_43-2.jpg")
rows, columns, channel=img.shape
rgb=np.zeros((rows, columns, channel), dtype=np.uint8)
for i in range(rows):
    for j in range(columns):
        rgb[i][j][0]=img[i][j][2]#red color
        rgb[i][j][1]=img[i][j][1]#green color
        rgb[i][j][2]=img[i][j][0]#blue color
plt.imshow(rgb)
plt.title("Image")
plt.axis("off")
plt.show()
