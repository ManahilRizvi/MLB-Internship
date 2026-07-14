import matplotlib.pyplot as plt
img=plt.imread("jpeg_43-2.jpg")
img=img.copy()#we copied the image because image can be only read-only 
totalSum=0#stores the sum of all pixel values
totalCount=0#counts total values

#if image's type is grayscale then
if len(img.shape)==2:
    minVal=img[0][0][0]
    maxVal=img[0][0][0]
    for row in img:#checking every row in image
        for pixel in row:#checking every pixel of row and each pixel has only 1 value
                value=int(value)#as image's datatype was uint8 so during addition it was 
                #overflowing so to avoid it we used int(value) to convert uint8 value to int
                totalSum+=value
                totalCount+=1
                if value<minVal:
                    minVal=value
                if value>maxVal:
                    maxVal=value

else:
    minVal=img[0][0][0]
    maxVal=img[0][0][0]
    for row in img:#checking every row in image
        for pixel in row:#checking every pixel of row
            for value in pixel:#checking RGB values of pixel
                value=int(value)#as image's datatype was uint8 so during addition it was 
                #overflowing so to avoid it we used int(value) to convert uint8 value to int
                totalSum+=value
                totalCount+=1
                if value<minVal:
                    minVal=value
                if value>maxVal:
                    maxVal=value

mean=totalSum/totalCount
print("Mean pixel intensity", mean)
print("Minimum pixel value", minVal)
print("Maximum pixel value", maxVal)