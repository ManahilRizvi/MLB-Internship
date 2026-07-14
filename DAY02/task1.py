import matplotlib.pyplot as plt
img=plt.imread("jpeg_43-2.jpg")#through this we will load image
#img.shape means that the information is in this format (rows, columns, channels)
#as rows=height and columns=width so shape[0] contains height and shape[1] contains width
h=img.shape[0]
w=img.shape[1]
#if image is grayscale then there will be only 1 channel and the format of img.shape will
#only have rows and columns not channel (rows, columns) but if it is not grayscale 
#then it will have this format (rows, columns, channels).
if len(img.shape)==2:
    channel=1
else:
    channel=img.shape[2]

dataType=img.dtype
pixels=h*w
print("Height: ", h)
print("Width: ", w)
print("Number of channels: ", channel)
print("Data type: ", dataType)
print("Total number of pixels: ", pixels)