import cv2
import matplotlib.pyplot as plt
import numpy as np
#for grayscale image
grayImg=cv2.imread("cat.jpeg", 0)
if grayImg is None:
    print("No image...")
else:
    #for gaussian noise 
    #mean=0 and standard deviation=20
    noise=np.random.normal(0, 20, grayImg.shape)
    floatGray=grayImg.astype(np.float32)#converting img to float
    grayNoise=floatGray+noise#adding noise
    grayNoise=np.clip(grayNoise, 0, 255)#keeping pixel values between 0 and 255
    grayNoise=grayNoise.astype(np.uint8)#converting to unsigned integer
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(grayImg, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(grayNoise, cmap="gray")
    plt.title("Gaussian(Grayscale) Noise")
    plt.axis("off")
    plt.show()
#for rgb image    
img=cv2.imread("dog.jpg")
if img is None:
    print("No image...")
else:
    rows, columns, channel=img.shape
    rgbImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=img[i][j][0]
            green=img[i][j][1]
            red=img[i][j][2]

            rgbImg[i][j][0]=red
            rgbImg[i][j][1]=green
            rgbImg[i][j][2]=blue
    #for gaussian noise 
    #mean=0 and standard deviation=20
    noise=np.random.normal(0, 20, rgbImg.shape)
    floatRgb=rgbImg.astype(np.float32)#converting img to float
    rgbNoise=floatRgb+noise#adding noise
    rgbNoise=np.clip(rgbNoise, 0, 255)#keeping pixel values between 0 and 255
    rgbNoise=rgbNoise.astype(np.uint8)#converting to unsigned integer
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgbImg)
    plt.title("RGB Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(rgbNoise)
    plt.title("Gaussian(RGB) Noise")
    plt.axis("off")
    plt.show()
    
#image noise is unwanted random variation in pixel values that 
#reduces quality of image. it makes image look grainy and distorted.
#it occurs due to these reasons:
#1. low light conditions while capturing image
#2. errors during image transmission or compression
#3. imperfections in camera sensor
#4. electronic interface during image acquisition
#5. high iso settings in camera
