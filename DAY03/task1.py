import cv2
import matplotlib.pyplot as plt
#reading image and storing in BGR format due to OpenCV
img=cv2.imread("orange-flower.jpg")

#as img.shape provides dimensions like in this format (rows, columns, channels) so we are stroing those in these variables
height, width, channel=img.shape
rgb=img.copy()#copying image

#visiting every pixel of image
for i in range(height):#loop of rows
    for j in range(width):#loop of columns
        #storing values of bgr by using indexing of channels
        blue=img[i, j, 0]#channel blue
        green=img[i, j, 1]#channel green
        red=img[i, j, 2]#channel red

        #swapping values of bgr image channels with rgb image channels
        rgb[i, j, 0]=red#red becomes first channel
        rgb[i, j, 1]=green#green always stay in middle
        rgb[i, j, 2]=blue#blue becomes last channel

#to show two images creating a figure and figsize controlling the  size of display window
plt.figure(figsize=(10, 5))

#dividing it into 2 plots
#subplot has format of this (rows, columns, position) which means that bgr image will be in first part 
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("BGR")#title of image
plt.axis("off")#hiding x-axis and y-axis numbers

#now rgb image will be in second part 
plt.subplot(1, 2, 2)
plt.imshow(rgb)
plt.title("RGB")#title of image
plt.axis("off")#hiding x-axis and y-axis numbers

plt.show()

#OpenCV reads images in BGR format because it was designed 
#to work with older image formats that uses BGR. That's why 
#OpenCV still uses BGR by default.