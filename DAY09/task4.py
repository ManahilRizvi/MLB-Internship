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

img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image...")
else:
    score=varianceOfLaplacian(img)
    print("Grayscale Image")
    print("Blur Score: ", score)
    if score>100:
        print("Result: Sharp Image")
    else:
        print("Result: Blurry Image")

img=cv2.imread("dog.jpg")
if img is None:
    print("No image...")
else:
    grayImg=rgbToGray(img)
    score=varianceOfLaplacian(grayImg)
    print("\nDog Image")
    print("Blur Score: ", score)
    if score>100:
        print("Result: Sharp Image")
    else:
        print("Result: Blurry Image")

img=cv2.imread("jpeg_43-2.jpg")
if img is None:
    print("No image...")
else:
    grayImg=rgbToGray(img)
    score=varianceOfLaplacian(grayImg)
    print("\nFlower Image")
    print("Blur Score: ", score)
    if score>100:
        print("Result: Sharp Image")
    else:
        print("Result: Blurry Image")