import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalization(image):
    rows, columns=image.shape
    image=np.abs(image)
    minVal=np.min(image)
    maxVal=np.max(image)
    result=np.zeros(image.shape, dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            if maxVal!=minVal:
                value=((image[i][j]-minVal)/(maxVal-minVal))*255
                if value>255:
                    value=255
                elif value<0:
                    value=0
                result[i][j]=int(value)
    return result

def sobel(image):
    rows, columns=image.shape
    sobelX=[[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]
    sobelY=[[-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]]

    gxImg=np.zeros((rows, columns), dtype=np.float32)
    gyImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
            for j in range(1, columns-1):
                sumX=0
                sumY=0
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        pixel=int(image[i+m][j+n])
                        sumX+=pixel*sobelX[m+1][n+1]
                        sumY+=pixel*sobelY[m+1][n+1]
                gxImg[i][j]=sumX
                gyImg[i][j]=sumY

    magnitudeImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(rows):
        for j in range(columns):
            magnitudeImg[i][j]=np.sqrt((gxImg[i][j]**2)+(gyImg[i][j]**2))

    gxDisplay=normalization(gxImg)
    gyDisplay=normalization(gyImg)
    magDisplay=normalization(magnitudeImg)
    return gxDisplay, gyDisplay, magDisplay

def laplacian(image):
    rows, columns=image.shape
    kernelLap=[[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]]
    lapImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(image[i+m][j+n])
                    value+=pixel*kernelLap[m+1][n+1]
            lapImg[i][j]=value
    lapDisplay=normalization(lapImg)
    return lapDisplay

def log(image):
    rows, columns=image.shape
    kernelGauss=[[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1]]

    gaussImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(image[i+m][j+n])
                    value+=pixel*kernelGauss[m+1][n+1]
            gaussImg[i][j]=value/16

    kernelLap=[[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]]
    logImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=gaussImg[i+m][j+n]
                    value+=pixel*kernelLap[m+1][n+1]
            logImg[i][j]=value
    gaussDisplay=normalization(gaussImg)
    logDisplay=normalization(logImg)
    return gaussDisplay, logDisplay

def DoGNorm(image):
    rows, columns=image.shape
    image=np.abs(image)
    maxVal=np.max(image)
    result=np.zeros(image.shape, dtype=np.uint8)
    if maxVal==0:
        return result
    for i in range(rows):
        for j in range(columns):
            value=(image[i][j]/maxVal)*255
            if value>255:
                value=255
            result[i][j]=int(value)
    return result 

def dog(image):
    rows, columns=image.shape
    kernel=[[1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]]

    gaussImg1=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(image[i+m][j+n])
                    value+=pixel*kernel[m+1][n+1]
            gaussImg1[i][j]=value/16

    gaussImg2=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(gaussImg1[i+m][j+n])
                    value+=pixel*kernel[m+1][n+1]
            gaussImg2[i][j]=value/16

    dogImg=gaussImg1-gaussImg2
    dogDisplay=DoGNorm(dogImg)
    gaussDisplay1=gaussImg1.astype(np.uint8)
    gaussDisplay2=gaussImg2.astype(np.uint8)
    return gaussDisplay1, gaussDisplay2, dogDisplay

def canny(image):
    rows, columns=image.shape
    kernelGauss=[[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]]

    gaussImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(grayImg[i+m][j+n])
                    value+=pixel*kernelGauss[m+1][n+1]
            gaussImg[i][j]=value/16

    kernelX=[[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]

    kernelY=[[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]]

    gradientX=np.zeros((rows, columns), dtype=np.float32)
    gradientY=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            gx=0
            gy=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=gaussImg[i+m][j+n]
                    gx+=pixel*kernelX[m+1][n+1]
                    gy+=pixel*kernelY[m+1][n+1]
            gradientX[i][j]=gx
            gradientY[i][j]=gy

    magGradient=np.zeros((rows, columns), dtype=np.float32)
    for i in range(rows):
        for j in range(columns):
            gx=gradientX[i][j]
            gy=gradientY[i][j]
            magnitude=np.sqrt((gx*gx)+(gy*gy))
            magGradient[i][j]=magnitude

    directionGrad=np.zeros((rows, columns), dtype=np.float32)
    for i in range(rows):
        for j in range(columns):
            gx=gradientX[i][j]
            gy=gradientY[i][j]
            angle=np.degrees(np.arctan2(gy, gx))
            if angle<0:
                angle+=180
            directionGrad[i][j]=angle

    nmsImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            angle=directionGrad[i][j]
            q=0
            r=0
            if (0<=angle<22.5) or (157.5<=angle<=180):
                q=magGradient[i][j+1]
                r=magGradient[i][j-1]

            elif 22.5<=angle<67.5:
                q=magGradient[i+1][j-1]
                r=magGradient[i-1][j+1]

            elif 67.5<=angle<112.5:
                q=magGradient[i+1][j]
                r=magGradient[i-1][j]

            elif 112.5<=angle<157.5:
                q=magGradient[i-1][j-1]
                r=magGradient[i+1][j+1]

            if (magGradient[i][j]>=q) and (magGradient[i][j]>=r):
                nmsImg[i][j]=magGradient[i][j]

            else:
                nmsImg[i][j]=0

    highThr=60
    lowThre=30
    strongPix=255
    weakPix=100
    thresholdImg=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            pixel=nmsImg[i][j]
            if pixel>=highThr:
                thresholdImg[i][j]=strongPix
            elif pixel>=lowThre:
                thresholdImg[i][j]=weakPix
            else:
                thresholdImg[i][j]=0

    cannyImg=thresholdImg.copy()
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            if cannyImg[i][j]==weakPix:
                connect=False
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        if cannyImg[i+m][j+n]==strongPix:
                            connect=True
                if connect:
                    cannyImg[i][j]=strongPix
                else:
                    cannyImg[i][j]=0

    gaussDisplay=gaussImg.astype(np.uint8)
    gradientDisplay=normalization(magGradient)
    nmsDisplay=normalization(nmsImg)
    return gaussDisplay, gradientDisplay, nmsDisplay, thresholdImg, cannyImg

img=cv2.imread("dog.jpg")
if img is None:
    print("No image...")
else:
    rows, columns, channel=img.shape
    grayImg=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=img[i][j][0]
            green=img[i][j][1]
            red=img[i][j][2]
            grayVal=int(0.114*blue+0.587*green+0.299*red)
            if grayVal>255:
                grayVal=255
            elif grayVal<0:
                grayVal=0
            grayImg[i][j]=grayVal
    gxImg, gyImg, sobelImg=sobel(grayImg)
    laplacianImg=laplacian(grayImg)
    gaussImg, logImg=log(grayImg)
    gaussImg1, gaussImg2, dogImg=dog(grayImg)
    gaussCanny, gradientImg, nmsImg, thresholdImg, cannyImg=canny(grayImg)

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(sobelImg, cmap="gray")
    plt.title("Sobel Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(laplacianImg, cmap="gray")
    plt.title("Laplacian Image")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(logImg, cmap="gray")
    plt.title("LoG Image")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(dogImg, cmap="gray")
    plt.title("DoG Image")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(cannyImg, cmap="gray")
    plt.title("Canny Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("-------EDGE DETECTION COMPARISON--------\n")
    print("1. Sobel")
    print("Strengths: It detects horizontal and vertical edges, simple and fast")
    print("Limitations: It is sensitive to noise and produces thick edges")
    print("Suitable Use Cases: Basic edge detection, Gradient calculation\n")

    print("2. Laplacian")
    print("Strengths: It detects edges in all directions")
    print("Limitations: It is highly sensitive to noise")
    print("Suitable Use Cases: Fine detail detection and image sharpening\n")

    print("3. LoG")
    print("Strengths: It removes noise before detecting edges")
    print("Limitations: It is slower than Laplacian")
    print("Suitable Use Cases: Noisy images and medical imaging\n")

    print("4. DoG")
    print("Strengths: It is faster approximation of LoG")
    print("Limitations: It depends on gaussian smoothing level")
    print("Suitable Use Cases: Feature detection and image segmentation\n")

    print("5. Canny")
    print("Strengths: It produces thin and accurate edges")
    print("Limitations: It is more computationally expensive")
    print("Suitable Use Cases: Object detection, computer vision and image analysis\n")