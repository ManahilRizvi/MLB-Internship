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

bitDepth=img.itemsize*8#calculated bit depth from img data type as itemsize gives us 
#size of one channel in bytes so we multiplied it by 8 to get bits
memBytes=h*w*channel*(bitDepth/8)#formula of calculating memory in bytes
memMB=memBytes/(1024*1024)
print("Memory in Bytes: ", memBytes)
print("Memory in MB: ", memMB)