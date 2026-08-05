import cv2
import numpy as np
import gradio as gr

def task1(image):
    if image is None:
        return None, None, None, None, None
    rows, columns, channel=image.shape
    grayImg=np.zeros((rows, columns), dtype=np.uint8)
    #converting rgb img to grayscale 
    for i in range(rows):
        for j in range(columns):
            #reading blue, green and red values
            blue=image[i][j][0]
            green=image[i][j][1]
            red=image[i][j][2]

            #calculating grayscale intensity using weighted formula
            grayVal=int(0.114*blue+0.587*green+0.299*red)

            #keeping pixel value within valid range
            if grayVal>255:
                grayVal=255
            elif grayVal<0:
                grayVal=0

            #storing grayscale value
            grayImg[i][j]=grayVal

#calculating gradient in X direction
#difference between right and left pixels is used to 
#detect vertical intensity changes
    gxImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            leftPix=int(grayImg[i][j-1])
            rightPix=int(grayImg[i][j+1])
            gxImg[i][j]=rightPix-leftPix

#calculating gradient in Y direction
#difference between bottom and top pixels is used to detect
#horizontal intensity changes
    gyImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            topPix=int(grayImg[i-1][j])
            bottomPix=int(grayImg[i+1][j])
            gyImg[i][j]=bottomPix-topPix

#calculating gradient magnitude
#combine X and Y gradients to measure overall edge strength
    magnitudeImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(rows):
        for j in range(columns):
            magnitudeImg[i][j]=np.sqrt((gxImg[i][j]**2)+(gyImg[i][j]**2))

#calculating gradient direction 
#find angle of gradient using arctangent of Gy and Gx
    directionImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(rows):
        for j in range(columns):
            directionImg[i][j]=np.arctan2(gyImg[i][j], gxImg[i][j])

##normalizing gradient X
#convert values into range 0 to 255 so they can be displayed as an image
    gxAbs=np.abs(gxImg)
    gxMin=np.min(gxAbs)
    gxMax=np.max(gxAbs)
    gxDisplay=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            if gxMax!=gxMin:
                value=((gxAbs[i][j]-gxMin)/(gxMax-gxMin))*255
                gxDisplay[i][j]=int(value)

#normalizing gradient Y
    gyAbs=np.abs(gyImg)
    gyMin=np.min(gyAbs)
    gyMax=np.max(gyAbs)
    gyDisplay=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            if gyMax!=gyMin:
                value=((gyAbs[i][j]-gyMin)/(gyMax-gyMin))*255
                gyDisplay[i][j]=int(value)

#normalizing gradient magnitude
    magMin=np.min(magnitudeImg)
    magMax=np.max(magnitudeImg)
    magDisplay=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            if magMax!=magMin:
                value=((magnitudeImg[i][j]-magMin)/(magMax-magMin))*255
                magDisplay[i][j]=int(value)

#normalizing gardient direction
#normalizing direction values only for visualization purpose
    dirMin=np.min(directionImg)
    dirMax=np.max(directionImg)
    dirDisplay=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            if dirMax!=dirMin:
                value=((directionImg[i][j]-dirMin)/(dirMax-dirMin))*255
                dirDisplay[i][j]=int(value)

    return (grayImg, gxDisplay, gyDisplay, magDisplay, dirDisplay)

def task2(image):
    if image is None:
        return None, None, None, None
    rows, columns, channel=image.shape
    def normalization(img):
        img=np.abs(img)
        minVal=np.min(img)
        maxVal=np.max(img)
        result=np.zeros(img.shape, dtype=np.uint8)
        for i in range(rows):
            for j in range(columns):
                if maxVal!=minVal:
                    value=((img[i][j]-minVal)/(maxVal-minVal))*255
                    result[i][j]=int(value)
        return result
    grayImg=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=image[i][j][0]
            green=image[i][j][1]
            red=image[i][j][2]
            grayVal=int(0.114*blue+0.587*green+0.299*red)

            if grayVal>255:
                grayVal=255
            elif grayVal<0:
                grayVal=0
            grayImg[i][j]=grayVal

    #defining sobel kernels
#sobel X detects vertical edges
#sobel Y detects horizontal edges
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
            #moving over 3x3 neighborhood
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(grayImg[i+m][j+n])

                    #multiplying neighboring pixels with sobel kernels
                    sumX+=pixel*sobelX[m+1][n+1]
                    sumY+=pixel*sobelY[m+1][n+1]

            #storing gradient values
            gxImg[i][j]=sumX
            gyImg[i][j]=sumY

    magnitudeImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(rows):
        for j in range(columns):
            magnitudeImg[i][j]=np.sqrt((gxImg[i][j]**2)+(gyImg[i][j]**2))

    gxDisplay=normalization(gxImg)
    gyDisplay=normalization(gyImg)
    magDisplay=normalization(magnitudeImg)
    return (grayImg, gxDisplay, gyDisplay, magDisplay)

def task3(image):
    if image is None:
        return None, None
    rows, columns, channel=image.shape
    def normlaiztion(img):
        img=np.abs(img)
        minVal=np.min(img)
        maxVal=np.max(img)
        result=np.zeros(img.shape, dtype=np.uint8)
        for i in range(rows):
            for j in range(columns):
                if maxVal!=minVal:
                    value=((img[i][j]-minVal)/(maxVal-minVal))*255
                    result[i][j]=int(value)
        return result
    grayImg=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=image[i][j][0]
            green=image[i][j][1]
            red=image[i][j][2]
            grayVal=int(0.114*blue+0.587*green+0.299*red)

            if grayVal>255:
                grayVal=255
            elif grayVal<0:
                grayVal=0
            grayImg[i][j]=grayVal

    kernelLap=[[0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]]

    laplacianImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            #moving over 3x3 neighborhood
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(grayImg[i+m][j+n])

                    #multiplying neighboring pixels with
                    #laplacian kernel
                    value+=pixel*kernelLap[m+1][n+1]

            #storing laplacian result
            laplacianImg[i][j]=value

    laplacianVal=np.abs(laplacianImg)
    laplacianDisplay=normlaiztion(laplacianVal)
    return (grayImg, laplacianDisplay)

def task4(image):
    if image is None:
        return None, None, None, None, None
    rows, columns, channel=image.shape
    def normalization(image):
        image=np.abs(image)
        minVal=np.min(image)
        maxVal=np.max(image)
        result=np.zeros(image.shape, dtype=np.uint8)
        for i in range(rows):
            for j in range(columns):
                if maxVal!=minVal:
                    value=((image[i][j]-minVal)/(maxVal-minVal))*255
                    result[i][j]=int(value)
        return result
    grayImg=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=image[i][j][0]
            green=image[i][j][1]
            red=image[i][j][2]
            grayVal=int(0.114*blue+0.587*green+0.299*red)

            if grayVal>255:
                grayVal=255
            elif grayVal<0:
                grayVal=0
            grayImg[i][j]=grayVal

    noise=np.random.normal(0, 20, (rows, columns))
    noiseImg=grayImg.astype(np.float32)+noise

    #keeping pixel values between 0 and 255
    for i in range(rows):
        for j in range(columns):
            if noiseImg[i][j]>255:
                noiseImg[i][j]=255
            elif noiseImg[i][j]<0:
                noiseImg[i][j]=0

    #connverting noisy image back to uint8
    noiseImg=noiseImg.astype(np.uint8)

    kernelGauss=[[1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]]

    gaussImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(noiseImg[i+m][j+n])
                    value+=pixel*kernelGauss[m+1][n+1]
            #dividing by 16 because sum of gaussian 
            #kernel is 16
            gaussImg[i][j]=value/16

    kernelLap=[[0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]]

    lapImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(noiseImg[i+m][j+n])
                    value+=pixel*kernelLap[m+1][n+1]
            lapImg[i][j]=value

    lapImg=np.abs(lapImg)
    lapDisplay=normalization(lapImg)

    lapImg=np.abs(lapImg)
    lapDisplay=normalization(lapImg)

    logImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=gaussImg[i+m][j+n]
                    value+=pixel*kernelLap[m+1][n+1]
            logImg[i][j]=value

    logImg=np.abs(logImg)
    gaussDisplay=normalization(gaussImg)
    logDisplay=normalization(logImg)
    for i in range(rows):
        for j in range(columns):
            value=int(logDisplay[i][j]*3)
            if value>255:
                value=255
            logDisplay[i][j]=value

    return (grayImg, noiseImg, lapDisplay, gaussDisplay, logDisplay)

def task5(image):
    if image is None:
        return None, None, None, None
    rows, columns, channel=image.shape
    def normalization(image):
        image=np.abs(image)
        minVal=np.min(image)
        maxVal=np.max(image)
        result=np.zeros(image.shape, dtype=np.uint8)
        for i in range(rows):
            for j in range(columns):
                if maxVal!=minVal:
                    value=((image[i][j]-minVal)/(maxVal-minVal))*255
                    result[i][j]=int(value)
        return result

    def DoGNorm(image):
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

    grayImg=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=image[i][j][0]
            green=image[i][j][1]
            red=image[i][j][2]
            grayVal=int(0.114*blue+0.587*green+0.299*red)

            if grayVal>255:
                grayVal=255
            elif grayVal<0:
                grayVal=0
            grayImg[i][j]=grayVal

    kernel=[[1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]]

    gaussImg1=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(grayImg[i+m][j+n])
                    value+=pixel*kernel[m+1][n+1]
            gaussImg1[i][j]=value/16

#secomd gaussian filter on already blurred image this
#produces stronger smoothing
    gaussImg2=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            value=0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    pixel=int(gaussImg1[i+m][j+n])
                    value+=pixel*kernel[m+1][n+1]
            gaussImg2[i][j]=value/16

#subtracting two blurred images 
#differenec shows edges and fine image details
    dogImg=gaussImg1-gaussImg2
    dogDisplay=DoGNorm(dogImg)
    gaussDisplay1=gaussImg1.astype(np.uint8)
    gaussDisplay2=gaussImg2.astype(np.uint8)
    return (grayImg, gaussDisplay1, gaussDisplay2, dogDisplay)

def task6(image):
    if image is None:
        return None, None, None, None, None, None
    img=image.copy()
    rows, columns, channel=img.shape
    def normalization(image):
        image=np.abs(image)
        minVal=np.min(image)
        maxVal=np.max(image)
        result=np.zeros(image.shape, dtype=np.uint8)
        for i in range(rows):
            for j in range(columns):
                if maxVal!=minVal:
                    value=((image[i][j]-minVal)/(maxVal-minVal))*255
                    result[i][j]=int(value)
        return result
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

#keeping only strongest edge pixel in gradient direction
#and remove all other neighboring pixels which makes edges thin
    nmsImg=np.zeros((rows, columns), dtype=np.float32)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            angle=directionGrad[i][j]
            q=0
            r=0
            #comparing left and right neighbors for horizontal edges
            if (0<=angle<22.5) or (157.5<=angle<=180):
                q=magGradient[i][j+1]
                r=magGradient[i][j-1]

            #comparing diagonal neighbors for 45 degree edges
            elif 22.5<=angle<67.5:
                q=magGradient[i+1][j-1]
                r=magGradient[i-1][j+1]

            #comparing top and bottom neighbors for vertical edges
            elif 67.5<=angle<112.5:
                q=magGradient[i+1][j]
                r=magGradient[i-1][j]

            #comparing other diagonal for 135 degree edges
            elif 112.5<=angle<157.5:
                q=magGradient[i-1][j-1]
                r=magGradient[i+1][j+1]

            #keeping only strongest pixel and supress
            #all weaker pixels
            if (magGradient[i][j]>=q) and (magGradient[i][j]>=r):
                nmsImg[i][j]=magGradient[i][j]

            else:
                nmsImg[i][j]=0

#classifying pixels into strong edges, weak edges and non edge pixels
    highThr=60
    lowThre=30
    strongPix=255
    weakPix=100
    thresholdImg=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            pixel=nmsImg[i][j]

            #strong edge
            if pixel>=highThr:
                thresholdImg[i][j]=strongPix
            #weak edge
            elif pixel>=lowThre:
                thresholdImg[i][j]=weakPix
            #not an edge
            else:
                thresholdImg[i][j]=0

#checking every weak edge pixel if it is connect to strong
#edge keep it otherwise remove it
    cannyImg=thresholdImg.copy()
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            if cannyImg[i][j]==weakPix:
                connect=False
                #checking all 8 neighboring pixels
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        if cannyImg[i+m][j+n]==strongPix:
                            connect=True

                #keep connected weak edges
                if connect:
                    cannyImg[i][j]=strongPix

                #removing alone weak edges
                else:
                    cannyImg[i][j]=0
    return (grayImg, gaussImg.astype(np.uint8), normalization(magGradient), normalization(nmsImg), thresholdImg, cannyImg)

def task7(image):
    if image is None:
        return None, None, None, None, None, None, ""
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
                        pixel=int(image[i+m][j+n])
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

    img=image
    rows, columns, channel=image.shape
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
    comparison = """-------EDGE DETECTION COMPARISON--------

    1. Sobel
    Strengths: It detects horizontal and vertical edges, simple and fast.
    Limitations: It is sensitive to noise and produces thick edges.
    Suitable Use Cases: Basic edge detection, Gradient calculation.

    2. Laplacian
    Strengths: It detects edges in all directions.
    Limitations: It is highly sensitive to noise.
    Suitable Use Cases: Fine detail detection and image sharpening.

    3. LoG
    Strengths: It removes noise before detecting edges.
    Limitations: It is slower than Laplacian.
    Suitable Use Cases: Noisy images and medical imaging.

    4. DoG
    Strengths: It is a faster approximation of LoG.
    Limitations: It depends on Gaussian smoothing level.
    Suitable Use Cases: Feature detection and image segmentation.

    5. Canny
    Strengths: It produces thin and accurate edges.
    Limitations: It is more computationally expensive.
    Suitable Use Cases: Object detection, computer vision and image analysis.
    """
    return(grayImg, sobelImg, laplacianImg, logImg, dogImg, cannyImg, comparison)

demo = gr.TabbedInterface(

    [
        gr.Interface(
            fn=task1,
            inputs=gr.Image(label="Upload Image"),
            outputs=[
                gr.Image(label="Grayscale Image"),
                gr.Image(label="Gradient X"),
                gr.Image(label="Gradient Y"),
                gr.Image(label="Gradient Magnitude"),
                gr.Image(label="Gradient Direction")
            ],
            title="Task 1: Image Gradient"
        ),

        gr.Interface(
            fn=task2,
            inputs=gr.Image(label="Upload Image"),
            outputs=[
                gr.Image(label="Grayscale Image"),
                gr.Image(label="Sobel X"),
                gr.Image(label="Sobel Y"),
                gr.Image(label="Sobel Magnitude")
            ],
            title="Task 2: Sobel Operator"
        ),

        gr.Interface(
            fn=task3,
            inputs=gr.Image(label="Upload Image"),
            outputs=[
                gr.Image(label="Grayscale Image"),
                gr.Image(label="Laplacian Edge Detection")
            ],
            title="Task 3: Laplacian"
        ),

        gr.Interface(
            fn=task4,
            inputs=gr.Image(label="Upload Image"),
            outputs=[
                gr.Image(label="Grayscale Image"),
                gr.Image(label="Noisy Image"),
                gr.Image(label="Laplacian"),
                gr.Image(label="Gaussian Blur"),
                gr.Image(label="LoG Image")
            ],
            title="Task 4: Laplacian of Gaussian"
        ),

        gr.Interface(
            fn=task5,
            inputs=gr.Image(label="Upload Image"),
            outputs=[
                gr.Image(label="Grayscale Image"),
                gr.Image(label="Gaussian Blur 1"),
                gr.Image(label="Gaussian Blur 2"),
                gr.Image(label="Difference of Gaussian")
            ],
            title="Task 5: Difference of Gaussian"
        ),

        gr.Interface(
            fn=task6,
            inputs=gr.Image(label="Upload Image"),
            outputs=[
                gr.Image(label="Grayscale Image"),
                gr.Image(label="Gaussian Smoothing"),
                gr.Image(label="Gradient Magnitude"),
                gr.Image(label="Non Maximum Suppression"),
                gr.Image(label="Double Thresholding"),
                gr.Image(label="Canny Edge Detection")
            ],
            title="Task 6: Canny Edge Detection"
        ),

        gr.Interface(
            fn=task7,
            inputs=gr.Image(label="Upload Image"),
            outputs=[
                gr.Image(label="Grayscale Image"),
                gr.Image(label="Sobel"),
                gr.Image(label="Laplacian"),
                gr.Image(label="LoG"),
                gr.Image(label="DoG"),
                gr.Image(label="Canny"),
                gr.Textbox(label="Comparison", lines=18)
            ],
            title="Task 7: Edge Detection Comparison"
        )

    ],

    tab_names=[
        "Task 1",
        "Task 2",
        "Task 3",
        "Task 4",
        "Task 5",
        "Task 6",
        "Task 7"
    ]
)

demo.launch()