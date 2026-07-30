import cv2
import numpy as np

def rgbToGray(img):
    rows, columns, channel=img.shape
    gray=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=img[i][j][0]
            green=img[i][j][1]
            red=img[i][j][2]
            gray[i][j]=int(0.114*blue+0.587*green+0.299*red)
    return gray

def varianceOfLaplacian(gray):
    rows, columns=gray.shape
    laplacianImg=np.zeros((rows, columns), dtype=np.float32)
    kernel=np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])

    for i in range(1, rows-1):
        for j in range(1, columns-1):
            total=0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    pixel=int(gray[i+x][j+y])
                    weight=kernel[x+1][y+1]
                    total+=pixel*weight
            laplacianImg[i][j]=total

    total=0
    product=rows*columns
    for i in range(rows):
        for j in range(columns):
            total+=laplacianImg[i][j]
    mean=total/product

    variance=0
    for i in range(rows):
        for j in range(columns):
            difference=laplacianImg[i][j]-mean
            variance+=difference*difference
    variance=variance/product
    return variance

images=["cat.jpeg",
        "clear.jpeg",
        "dog.jpg",
        "images.jpeg",
        "jpeg_43-2.jpg",
        "lowCon.jpg",
        "mouse.jpeg",
        "orange-flower.jpg",
        "river.jpeg",
        "road.jpeg"]
print("Image Name\Variance of Laplacian Score\tBlur Level (Sharp/Blurry)\tObservation")
print("---------------------------------------------------------------------------------------------")
thresholdVal=100
for image in images:
    img=cv2.imread(image)
    grayImg=rgbToGray(img)
    blurScore=varianceOfLaplacian(grayImg)
    if blurScore>thresholdVal:
        level="Sharp Image"
        observation="Edges are CLEAR"
    else:
        level="Blurry Image"
        observation="Edges are not CLEAR"
    print(image, "\t\t", round(blurScore, 2), "\t\t", level, "\t\t", observation)


#images with higher variance of laplacian scores contain more edges
#information that's why they are classified as sharp images.
#images with lower variance of laplacian scores have fewer edges
#and details that's why they are classified as blurry images.

#river.jpeg has highest score(1068.55) so it is sharpest imafe
#clear.jpeg has lowest score(19.83) so it is blurriest image
#images.jpeg and orange-flower.jpg also have low scores and 
#classified as blurred images
#most other images have scores above threshold (100) which indicates good 
#good edge details.
