import matplotlib.pyplot as plt
img=plt.imread("jpeg_43-2.jpg")
img=img.copy()#we copied the image because image can be only read-only 
totalPix=0
blackPix=0
whitePix=0
abovePix=0

#if image's type is grayscale then
if len(img.shape)==2:
    for row in img:#checking every row in image
        for pixel in row:#checking every pixel of row and each pixel has only 1 value
            value=int(pixel)#as image's datatype was uint8 so during addition it was 
            #overflowing so to avoid it we used int(value) to convert uint8 value to int
            totalPix+=1
            if value==0:
                blackPix+=1
            if value==255:
                whitePix+=1
            if value>200:
                abovePix+=1
else:
    for row in img:#checking every row in image
        for pixel in row:#checking every pixel of row and each pixel has only 1 value
            totalPix+=1
            #storing red, green and blue values 
            R=int(pixel[0])
            G=int(pixel[1])
            B=int(pixel[2])
            if R==0 and G==0 and B==0:
                blackPix+=1
            if R==255 and G==255 and B==255:
                whitePix+=1
            
            intensity=(R+G+B)/3
            if intensity>200:
                abovePix+=1

print("Total pixels: ", totalPix)
print("Black pixels: ", blackPix)
print("White pixels: ", whitePix)
print("Pixels with intensity > 200: ", abovePix)