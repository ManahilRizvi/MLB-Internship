import matplotlib.pyplot as plt
img=plt.imread("jpeg_43-2.jpg")
img=img.copy()#we copied the image because image can be only read-only 
if len(img.shape)!=3:#if it is grayscale
    print("Not an RGB image...")
else:
    R=img.copy()
    G=img.copy()
    B=img.copy()
    for row in R:#for red channel
        for pixel in row:
            pixel[1]=0#removing green
            pixel[2]=0#removing blue
    for row in G:#for green channel
        for pixel in row:
            pixel[0]=0#removing red
            pixel[2]=0#removing blue
    for row in B:#for blue channel
        for pixel in row:
            pixel[0]=0#removing red
            pixel[1]=0#removing green
    plt.imsave("RedChannel.jpg", R)
    plt.imsave("GreenChannel.jpg", G)
    plt.imsave("BlueChannel.jpg", B)
    print("DONE SAVING...")