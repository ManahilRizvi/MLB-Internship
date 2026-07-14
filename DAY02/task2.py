import matplotlib.pyplot as plt
img=plt.imread("jpeg_43-2.jpg")
img=img.copy()
#we copied the image because image can be only read-only 
img[10][10]=[255, 255, 255]#color white
img[20][20]=[0, 255, 255]#color cyan
img[30][30]=[0, 0, 255]#color blue
img[40][40]=[0, 0, 0]#color black 
img[50][50]=[255, 255, 0]#color yellow
plt.imsave("new_image.jpg", img)
print("Image Saved...")