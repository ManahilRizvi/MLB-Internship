import cv2
import gradio as gr
import numpy as np
import os

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

def task1(image):
    if image is None:
        return None, None
    #for grayscale image
    grayImg=np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    rows, columns=grayImg.shape
    for i in range(rows):
        for j in range(columns):
            blue=image[i][j][0]
            green=image[i][j][1]
            red=image[i][j][2]
            gray=int(0.114*blue+0.587*green+0.299*red)
            grayImg[i][j]=gray
        #for gaussian noise 
        #mean=0 and standard deviation=20
    noise=np.random.normal(0, 20, grayImg.shape)
    floatGray=grayImg.astype(np.float32)#converting img to float
    grayNoise=floatGray+noise#adding noise
    grayNoise=np.clip(grayNoise, 0, 255)#keeping pixel values between 0 and 255
    grayNoise=grayNoise.astype(np.uint8)

    rows, columns, channel=image.shape
    rgbImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    rgbImg=image.copy()
    #for gaussian noise 
    #mean=0 and standard deviation=20
    noise=np.random.normal(0, 20, rgbImg.shape)
    floatRgb=rgbImg.astype(np.float32)#converting img to float
    rgbNoise=floatRgb+noise#adding noise
    rgbNoise=np.clip(rgbNoise, 0, 255)#keeping pixel values between 0 and 255
    rgbNoise=rgbNoise.astype(np.uint8)#converting to unsigned integer
    return grayNoise, rgbNoise

def task2(image):
    if image is None:
        return None, None, None, None, None, None

    rows, columns, channel=image.shape
    grayImg=np.zeros((rows, columns), dtype=np.uint8)

    for i in range(rows):
        for j in range(columns):
            blue=image[i][j][0]
            green=image[i][j][1]
            red=image[i][j][2]
            gray=int(0.114*blue+0.587*green+0.299*red)
            grayImg[i][j]=gray

    noise=np.random.normal(0, 20, grayImg.shape)
    floatGray=grayImg.astype(np.float32)#converting img to float
    grayNoise=floatGray+noise#adding noise
    grayNoise=np.clip(grayNoise, 0, 255)#keeping pixel values between 0 and 255
    grayNoise=grayNoise.astype(np.uint8)
    rows, columns=grayNoise.shape

    meanGray=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            total=0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    total+=int(grayNoise[i+x][j+y])
            meanGray[i][j]=total//9

    gaussianGray=np.zeros((rows, columns), dtype=np.uint8)
    kernel=[[1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]]
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            total=0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    pixel=int(grayNoise[i+x][j+y])
                    weight=kernel[x+1][y+1]
                    total+=pixel*weight
            gaussianGray[i][j]=total//16

    rows, columns, channel=image.shape
    rgbImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    rgbImg=image.copy()
    noise=np.random.normal(0, 20, rgbImg.shape)
    floatRgb=rgbImg.astype(np.float32)#converting img to float
    rgbNoise=floatRgb+noise#adding noise
    rgbNoise=np.clip(rgbNoise, 0, 255)#keeping pixel values between 0 and 255
    rgbNoise=rgbNoise.astype(np.uint8)
    rows, columns, channel=rgbNoise.shape

    meanRgb=np.zeros((rows, columns, channel), dtype=np.uint8)
    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        total+=int(rgbNoise[i+x][j+y][k])
                meanRgb[i][j][k]=total//9

    gaussianRgb=np.zeros((rows, columns, channel), dtype=np.uint8)
    kernel=[[1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]]
    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        pixel=int(rgbNoise[i+x][j+y][k])
                        weight=kernel[x+1][y+1]
                        total+=pixel*weight
                gaussianRgb[i][j][k]=total//16
    return (grayNoise, meanGray, gaussianGray,
            rgbNoise, meanRgb, gaussianRgb)

def task3(image):
    if image is None:
        return None, None, None, None, None, None, None, None, None, None
    rows, columns, channel=image.shape
    img=np.zeros((rows, columns), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=image[i][j][0]
            green=image[i][j][1]
            red=image[i][j][2]
            gray=int(0.114*blue+0.587*green+0.299*red)
            img[i][j]=gray
    rows, columns=img.shape
    gaussianImg3=img.copy()
    gaussKernel3=np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]])
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            total=0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    pixel=int(img[i+x][j+y])
                    weight=gaussKernel3[x+1][y+1]
                    total+=pixel*weight
            gaussianImg3[i][j]=total//16

    gaussianImg5=img.copy()
    gaussKernel5=np.array([[1, 4, 6, 4, 1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4, 6, 4, 1]])
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            total=0
            for x in range(-2, 3):
                for y in range(-2, 3):
                    pixel=int(img[i+x][j+y])
                    weight=gaussKernel5[x+2][y+2]
                    total+=pixel*weight
            gaussianImg5[i][j]=total//256

    gaussianImg7=img.copy()
    gaussKernel7=np.array([[0, 0, 1, 2, 1, 0, 0],
                            [0, 3, 13, 22, 13, 3, 0],
                            [1, 13, 59, 97, 59, 13, 1],
                            [2, 22, 97, 159, 97, 22, 2],
                            [1, 13, 59, 97, 59, 13, 1],
                            [0, 3, 13, 22, 13, 3, 0],
                            [0, 0, 1, 2, 1, 0, 0]])
    sum=np.sum(gaussKernel7)
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            total=0
            for x in range(-3, 4):
                for y in range(-3, 4):
                    pixel=int(img[i+x][j+y])
                    weight=gaussKernel7[x+3][y+3]
                    total+=pixel*weight
            gaussianImg7[i][j]=total//sum

    motionImg3=img.copy()
    motionKernel3=np.array([[0, 0, 0],
                            [1, 1, 1],
                            [0, 0, 0]])
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            total=0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    pixel=int(img[i+x][j+y])
                    weight=motionKernel3[x+1][y+1]
                    total+=pixel*weight
            motionImg3[i][j]=total//3

    motionImg5=img.copy()
    motionKernel5=np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            total=0
            for x in range(-2, 3):
                for y in range(-2, 3):
                    pixel=int(img[i+x][j+y])
                    weight=motionKernel5[x+2][y+2]
                    total+=pixel*weight
            motionImg5[i][j]=total//5

    motionImg7=img.copy()
    motionKernel7=np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]])
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            total=0
            for x in range(-3, 4):
                for y in range(-3, 4):
                    pixel=int(img[i+x][j+y])
                    weight=motionKernel7[x+3][y+3]
                    total+=pixel*weight
            motionImg7[i][j]=total//7

    medianImg3=img.copy()
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            values=[]
            for x in range(-1, 2):
                for y in range(-1, 2):
                    values.append(int(img[i+x][j+y]))
            for a in range(len(values)):
                for b in range(len(values)-1):
                    if values[b]>values[b+1]:
                        temp=values[b]
                        values[b]=values[b+1]
                        values[b+1]=temp
            medianImg3[i][j]=values[4]

    medianImg5=img.copy()
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            values=[]
            for x in range(-2, 3):
                for y in range(-2, 3):
                    values.append(int(img[i+x][j+y]))
            for a in range(len(values)):
                for b in range(len(values)-1):
                    if values[b]>values[b+1]:
                        temp=values[b]
                        values[b]=values[b+1]
                        values[b+1]=temp
            medianImg5[i][j]=values[12]

    medianImg7=img.copy()
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            values=[]
            for x in range(-3, 4):
                for y in range(-3, 4):
                    values.append(int(img[i+x][j+y]))
            for a in range(len(values)):
                for b in range(len(values)-1):
                    if values[b]>values[b+1]:
                        temp=values[b]
                        values[b]=values[b+1]
                        values[b+1]=temp
            medianImg7[i][j]=values[24]

    return(gaussianImg3, gaussianImg5, gaussianImg7,
           motionImg3, motionImg5, motionImg7,
           medianImg3, medianImg5, medianImg7)

def task4(image):
    if image is None:
        return None, "No image..."

    if len(image.shape)==2:
        grayImg=image
    else:
        grayImg=rgbToGray(image)
    thresholdVal=100
    blurScore=varianceOfLaplacian(grayImg)
    if blurScore>thresholdVal:
        result="Sharp Image" 
    else:
        result="BLurry Image" 
    return grayImg, f"Blur Score: {blurScore:.2f}", result

def task5(images):
    if images is None:
        return "No images..."

    thresholdVal=100
    result=f"{'Image Name':<20}{'Score':<15}{'Blur Level':<18}Observation\n"
    result+="-"*70+"\n"
    for image in images:
        img=cv2.imread(image.name)
        if img is None:
            continue
        if len(img.shape)==2:
            grayImg=img
        else:
            grayImg=rgbToGray(img)
        blurScore=varianceOfLaplacian(grayImg)
        if blurScore > thresholdVal:
            level = "Sharp"
            observation = "Edges are CLEAR"
        else:
            level = "Blurry"
            observation = "Edges are NOT CLEAR"
        filename = os.path.basename(image.name)
        result+=f"{filename:<20}{blurScore:<15.2f}{level:<18}{observation}\n"
    return result

with gr.Blocks(title="Module 7&8: Noise and Blur") as demo:

    gr.Markdown("# Module 7: Noise in Images")
    with gr.Tab("Task 1: Gaussian Noise"):

        inputImage = gr.Image(label="Upload Image")

        btn = gr.Button("Generate Noise")

        grayOutput = gr.Image(label="Gaussian Noise (Gray)")
        rgbOutput = gr.Image(label="Gaussian Noise (RGB)")

        btn.click(
            fn=task1,
            inputs=inputImage,
            outputs=[grayOutput, rgbOutput]
        )

    with gr.Tab("Task 2: Noise Removal"):

        inputImage = gr.Image(label="Upload Image")

        btn = gr.Button("Apply Filters")

        grayNoise = gr.Image(label="Gray Noise")
        grayMean = gr.Image(label="Mean Filter (Gray)")
        grayGaussian = gr.Image(label="Gaussian Filter (Gray)")

        rgbNoise = gr.Image(label="RGB Noise")
        rgbMean = gr.Image(label="Mean Filter (RGB)")
        rgbGaussian = gr.Image(label="Gaussian Filter (RGB)")

        btn.click(
            fn=task2,
            inputs=inputImage,
            outputs=[
                grayNoise,
                grayMean,
                grayGaussian,
                rgbNoise,
                rgbMean,
                rgbGaussian
            ]
        )
    gr.Markdown("# Module 8: Blur")
    with gr.Tab("Task 3: Blur Generation"):

        inputImage = gr.Image(label="Upload Image")

        btn = gr.Button("Generate Blur")

        g3 = gr.Image(label="Gaussian 3x3")
        g5 = gr.Image(label="Gaussian 5x5")
        g7 = gr.Image(label="Gaussian 7x7")

        m3 = gr.Image(label="Motion 3x3")
        m5 = gr.Image(label="Motion 5x5")
        m7 = gr.Image(label="Motion 7x7")

        md3 = gr.Image(label="Median 3x3")
        md5 = gr.Image(label="Median 5x5")
        md7 = gr.Image(label="Median 7x7")

        btn.click(
            fn=task3,
            inputs=inputImage,
            outputs=[
                g3,g5,g7,
                m3,m5,m7,
                md3,md5,md7
            ]
        )

    with gr.Tab("Task 4: Blur Detection"):

        inputImage = gr.Image(label="Upload Image")

        btn = gr.Button("Detect Blur")

        gray = gr.Image(label="Grayscale")

        score = gr.Textbox(label="Blur Score")

        result = gr.Textbox(label="Result")

        btn.click(
            fn=task4,
            inputs=inputImage,
            outputs=[
                gray,
                score,
                result
            ]
        )

    with gr.Tab("Task 5: Blur Metrics"):

        inputFiles = gr.File(
            file_count="multiple",
            file_types=["image"],
            label="Upload Images"
        )

        btn = gr.Button("Analyze Images")

        table = gr.Textbox(
            label="Blur Analysis",
            lines=15
        )

        btn.click(
            fn=task5,
            inputs=inputFiles,
            outputs=table
        )
demo.launch()