import matplotlib.pyplot as plt
img=plt.imread("jpeg_43-2.jpg")
gray=img.copy()#we copied the image because image can be only read-only 
if len(img.shape)!=3:#if it is grayscale
    print("Already GRAYSCALE...")
else:
    for row in gray:#checking every row in image
        for pixel in row:#checking every pixel of row
            #storing red, green and blue values 
            R=int(pixel[0])
            G=int(pixel[1])
            B=int(pixel[2])
            #formula of converting RGB into Grayscale
            intensity=int(0.299*R+0.587*G+0.114*B)
            #as values of red, green and blue are equal in grayscale so we are assigning 
            #same intensity
            pixel[0]=intensity
            pixel[1]=intensity
            pixel[2]=intensity
    plt.imsave("GrayscaleImg.jpg", gray)
    print("Done...")