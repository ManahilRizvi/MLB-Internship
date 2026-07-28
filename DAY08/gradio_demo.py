import gradio as gr
import cv2
import numpy as np

#task1
def task1(image, brightness):
    img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    rows, columns, channel=img.shape
    rgbImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=img[i][j][0]
            green=img[i][j][1]
            red=img[i][j][2]

            rgbImg[i][j][0]=red
            rgbImg[i][j][1]=green
            rgbImg[i][j][2]=blue

    brightnessVal=int(brightness)
    outputImg=rgbImg.copy()
    for i in range(rows):
        for j in range(columns):
            for k in range(channel):
                value=int(rgbImg[i][j][k])+brightnessVal
                if value>255:
                    value=255
                elif value<0:
                    value=0
                outputImg[i][j][k]=value
    return outputImg

def task2(image, alpha, beta):
    img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    rows, columns, channel=img.shape
    rgbImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=img[i][j][0]
            green=img[i][j][1]
            red=img[i][j][2]

            rgbImg[i][j][0]=red
            rgbImg[i][j][1]=green
            rgbImg[i][j][2]=blue

    contrastImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            for k in range(channel):
                value=int(alpha*rgbImg[i][j][k]+beta)
                if value>255:
                    value=255
                elif value<0:
                    value=0
                contrastImg[i][j][k]=value
    return contrastImg

def task3(image):
    img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    rows, columns, channel=img.shape
    rgbImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=img[i][j][0]
            green=img[i][j][1]
            red=img[i][j][2]

            rgbImg[i][j][0]=red
            rgbImg[i][j][1]=green
            rgbImg[i][j][2]=blue

    kernel=np.array([[0, -1, 0], 
                    [-1, 5, -1],
                    [0, -1, 0]])
    sharpenImg=np.zeros((rows, columns, channel), dtype=np.uint8)

    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        total+=rgbImg[i+x][j+y][k]*kernel[x+1][y+1]
                if total>255:
                    total=255
                elif total<0:
                    total=0
                sharpenImg[i][j][k]=total
    return sharpenImg

def task4(image):
    img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    rows, columns, channel=img.shape
    rgbImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=img[i][j][0]
            green=img[i][j][1]
            red=img[i][j][2]

            rgbImg[i][j][0]=red
            rgbImg[i][j][1]=green
            rgbImg[i][j][2]=blue
    
    saltNoise=rgbImg.copy()
    pepperNoise=rgbImg.copy()
    saltPepperNoise=rgbImg.copy()
    gaussianNoise=rgbImg.copy()
    probability=0.05

    for i in range(1, rows-1):
        for j in range(1, columns-1):
            if np.random.rand()<probability:
                for k in range(channel):
                    saltNoise[i][j][k]=255

    for i in range(1, rows-1):
        for j in range(1, columns-1):
            if np.random.rand()<probability:
                for k in range(channel):
                    pepperNoise[i][j][k]=0

    for i in range(1, rows-1):
        for j in range(1, columns-1):
            randomVal=np.random.rand()
            if randomVal<probability:
                for k in range(channel):
                    saltPepperNoise[i][j][k]=0
            elif randomVal>(1-probability):
                for k in range(channel):
                    saltPepperNoise[i][j][k]=255

    meanVal=0
    sigmaVal=25
    for i in range(rows):
        for j in range(columns):
            for k in range(channel):
                noise=np.random.normal(meanVal, sigmaVal)
                value=int(rgbImg[i][j][k]+noise)
                if value>255:
                    value=255
                elif value<0:
                    value=0
                gaussianNoise[i][j][k]=value

    return saltNoise, pepperNoise, saltPepperNoise, gaussianNoise

def task5(image):
    img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    rows, columns, channel=img.shape
    rgbImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=img[i][j][0]
            green=img[i][j][1]
            red=img[i][j][2]

            rgbImg[i][j][0]=red
            rgbImg[i][j][1]=green
            rgbImg[i][j][2]=blue
    
    noiseImg=rgbImg.copy()
    probability=0.03

    for i in range(rows):
        for j in range(columns):
            randomVal=np.random.rand()
            if randomVal<probability:
                for k in range(channel):
                    noiseImg[i][j][k]=0
            elif randomVal>(1-probability):
                for k in range(channel):
                    noiseImg[i][j][k]=255

    meanImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    gaussianImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    medianImg=np.zeros((rows, columns, channel), dtype=np.uint8)

    kernelMean=np.ones((3, 3))
    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        total+=noiseImg[i+x][j+y][k]*kernelMean[x+1][y+1]
                meanImg[i][j][k]=total//9

    kernelGau=np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]])
    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        total+=noiseImg[i+x][j+y][k]*kernelGau[x+1][y+1]
                gaussianImg[i][j][k]=total//16

    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                values=[]
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        values.append(noiseImg[i+x][j+y][k])
                a=len(values)
                for m in range(a):
                    for n in range(0, a-m-1):
                        if values[n]>values[n+1]:
                            temp=values[n]
                            values[n]=values[n+1]
                            values[n+1]=temp

                medianImg[i][j][k]=values[4]
    return noiseImg, meanImg, gaussianImg, medianImg

def task6(image):
    img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    rows, columns, channel=img.shape
    rgbImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=img[i][j][0]
            green=img[i][j][1]
            red=img[i][j][2]

            rgbImg[i][j][0]=red
            rgbImg[i][j][1]=green
            rgbImg[i][j][2]=blue

    kernelBlur=np.ones((3, 3))
    blurImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        total+=rgbImg[i+x][j+y][k]*kernelBlur[x+1][y+1]
                blurImg[i][j][k]=total//9

    kernelSharp=np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
    restoreImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        total+=blurImg[i+x][j+y][k]*kernelSharp[x+1][y+1]
                if total>255:
                    total=255
                elif total<0:
                    total=0
                restoreImg[i][j][k]=total
    return rgbImg, blurImg, restoreImg

def task7(image):
    img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    rows, columns, channel=img.shape
    rgbImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=img[i][j][0]
            green=img[i][j][1]
            red=img[i][j][2]

            rgbImg[i][j][0]=red
            rgbImg[i][j][1]=green
            rgbImg[i][j][2]=blue
    
    brightnessVal=50
    brighterImg=rgbImg.copy()

    for i in range(rows):
        for j in range(columns):
            for k in range(channel):
                value=int(rgbImg[i][j][k])+brightnessVal
                if value>255:
                    value=255
                elif value<0:
                    value=0
                brighterImg[i][j][k]=value

    alphaVal=1.5
    betaVal=0
    contrastImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(columns):
            for k in range(channel):
                value=int(alphaVal*rgbImg[i][j][k]+betaVal)
                if value>255:
                    value=255
                elif value<0:
                    value=0
                contrastImg[i][j][k]=value

    kernelSharp=np.array([[0, -1, 0], 
                            [-1, 5, -1],
                            [0, -1, 0]])
    sharpenImg=np.zeros((rows, columns, channel), dtype=np.uint8)

    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                total=0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        total+=rgbImg[i+x][j+y][k]*kernelSharp[x+1][y+1]
                if total>255:
                    total=255
                elif total<0:
                    total=0
                sharpenImg[i][j][k]=total

    noiseImg=rgbImg.copy()
    probability=0.03

    for i in range(rows):
        for j in range(columns):
            randomVal=np.random.rand()
            if randomVal<probability:
                for k in range(channel):
                    noiseImg[i][j][k]=0
            elif randomVal>(1-probability):
                for k in range(channel):
                    noiseImg[i][j][k]=255

    denoiseImg=np.zeros((rows, columns, channel), dtype=np.uint8)
    for k in range(channel):
        for i in range(1, rows-1):
            for j in range(1, columns-1):
                values=[]
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        values.append(noiseImg[i+x][j+y][k])
                a=len(values)
                for m in range(a):
                    for n in range(0, a-m-1):
                        if values[n]>values[n+1]:
                            temp=values[n]
                            values[n]=values[n+1]
                            values[n+1]=temp

                denoiseImg[i][j][k]=values[4]
    return brighterImg, contrastImg, sharpenImg, noiseImg, denoiseImg

with gr.Blocks(title="Module 6 : Image Enhancement") as demo:
    gr.Markdown("# Module 6 : Image Enhancement")
    with gr.Tab("Task 1 : Brightness Adjustment"):

        input1 = gr.Image(
            type="numpy",
            label="Upload Image"
        )

        brightness = gr.Slider(
            minimum=-100,
            maximum=100,
            value=50,
            step=1,
            label="Brightness"
        )

        output1 = gr.Image(
            label="Brightness Adjusted Image"
        )

        button1 = gr.Button("Apply")

        button1.click(
            fn=task1,
            inputs=[input1, brightness],
            outputs=output1
        )

    with gr.Tab("Task 2 : Contrast Adjustment"):

        input2 = gr.Image(
            type="numpy",
            label="Upload Image"
        )

        alpha = gr.Slider(
            minimum=0.1,
            maximum=3.0,
            value=1.5,
            step=0.1,
            label="Alpha (Contrast)"
        )

        beta = gr.Slider(
            minimum=-100,
            maximum=100,
            value=0,
            step=1,
            label="Beta (Brightness Offset)"
        )

        output2 = gr.Image(
            label="Contrast Adjusted Image"
        )

        button2 = gr.Button("Apply")

        button2.click(
            fn=task2,
            inputs=[input2, alpha, beta],
            outputs=output2
        )

    with gr.Tab("Task 3 : Image Sharpening"):

        input3 = gr.Image(
            type="numpy",
            label="Upload Image"
        )

        output3 = gr.Image(
            label="Sharpened Image"
        )

        button3 = gr.Button("Apply")

        button3.click(
            fn=task3,
            inputs=input3,
            outputs=output3
        )

    with gr.Tab("Task 4 : Noise Generation"):

        input4 = gr.Image(
            type="numpy",
            label="Upload Image"
        )

        output41 = gr.Image(
            label="Salt Noise"
        )

        output42 = gr.Image(
            label="Pepper Noise"
        )

        output43 = gr.Image(
            label="Salt & Pepper Noise"
        )

        output44 = gr.Image(
            label="Gaussian Noise"
        )

        button4 = gr.Button("Apply")

        button4.click(
            fn=task4,
            inputs=input4,
            outputs=[
                output41,
                output42,
                output43,
                output44
            ]
        )

    with gr.Tab("Task 5 : Noise Reduction"):

        input5 = gr.Image(
            type="numpy",
            label="Upload Image"
        )

        output51 = gr.Image(
            label="Noisy Image"
        )

        output52 = gr.Image(
            label="Mean Filter"
        )

        output53 = gr.Image(
            label="Gaussian Filter"
        )

        output54 = gr.Image(
            label="Median Filter"
        )

        button5 = gr.Button("Apply")

        button5.click(
            fn=task5,
            inputs=input5,
            outputs=[
                output51,
                output52,
                output53,
                output54
            ]
        )

    with gr.Tab("Task 6 : Deblurring Basics"):

        input6 = gr.Image(
            type="numpy",
            label="Upload Image"
        )

        output61 = gr.Image(
            label="Original Image"
        )

        output62 = gr.Image(
            label="Blurred Image"
        )

        output63 = gr.Image(
            label="Restored Image"
        )

        button6 = gr.Button("Apply")

        button6.click(
            fn=task6,
            inputs=input6,
            outputs=[
                output61,
                output62,
                output63
            ]
        )

    with gr.Tab("Task 7 : Image Enhancement Comparison"):

        input7 = gr.Image(
            type="numpy",
            label="Upload Image"
        )

        output71 = gr.Image(
            label="Brightness Adjustment"
        )

        output72 = gr.Image(
            label="Contrast Enhancement"
        )

        output73 = gr.Image(
            label="Sharpened Image"
        )

        output74 = gr.Image(
            label="Noisy Image"
        )

        output75 = gr.Image(
            label="Denoised Image"
        )

        button7 = gr.Button("Apply")

        button7.click(
            fn=task7,
            inputs=input7,
            outputs=[
                output71,
                output72,
                output73,
                output74,
                output75
            ]
        )

demo.launch()