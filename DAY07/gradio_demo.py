import cv2
import time
import numpy as np
import gradio as gr

def mean_filter(img):
    rows, columns=img.shape
    filterImg=np.zeros((rows, columns), dtype=np.uint8)
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            #calculating sum of all 9 neighboring pixels
            total=(
                int(img[i-1][j-1])+
                int(img[i-1][j])+
                int(img[i-1][j+1])+
                int(img[i][j-1])+
                int(img[i][j])+
                int(img[i][j+1])+
                int(img[i+1][j-1])+
                int(img[i+1][j])+
                int(img[i+1][j+1])
            )
            #calculating average
            average=total//9
            filterImg[i][j]=average
    return filterImg

def box_filter(img):
    rows, columns=img.shape
    filterImg=np.zeros((rows, columns), dtype=np.uint8)
    #box filter kernel
    kernel=np.array([[1, 1, 1,], [1, 1, 1,], [1, 1, 1,]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            filterImg[i][j]=totalSum//9
    return filterImg

def gaussian_filter(img):
    rows, columns=img.shape
    filterImg=np.zeros((rows, columns), dtype=np.uint8)
    #box filter kernel
    kernel=np.array([[1, 2, 1,], [2, 4, 2,], [1, 2, 1,]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing by dividing with sum of kernel values (16)
            filterImg[i][j]=totalSum//16
    return filterImg

def median_filter(img):
    rows, columns=img.shape
    noiseImg=img.copy()
    #adding salt and pepper noise
    for i in range(0, rows, 20):
        for j in range(0, columns, 20):
            noiseImg[i][j]=255#adding salt noise
            if i+10<rows and j+10<columns:
                noiseImg[i+10][j+10]=0
    filterImg=np.zeros((rows, columns), dtype=np.uint8)
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            pixels=[]#storing neighboring pixels
            #collecting all 9 neighboring pixels
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(noiseImg[i+m-1][j+n-1])#getting corresponding image pixel
                    pixels.append(pixel)
            #bubble sorting
            for x in range(len(pixels)-1):
                for y in range(len(pixels)-x-1):
                    if pixels[y]>pixels[y+1]:
                        temp=pixels[y]
                        pixels[y]=pixels[y+1]
                        pixels[y+1]=temp
            median=pixels[4]
            filterImg[i][j]=median
    return filterImg

def comparison_filters(img):
    rows, columns=img.shape
    noiseImg=img.copy()
    #adding salt and pepper noise
    for i in range(0, rows, 20):
        for j in range(0, columns, 20):
            noiseImg[i][j]=255#adding salt noise
            if i+10<rows and j+10<columns:
                noiseImg[i+10][j+10]=0
    filterImg=np.zeros((rows, columns), dtype=np.uint8)
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            #calculating sum of all 9 neighboring pixels
            total=(
                int(noiseImg[i-1][j-1])+
                int(noiseImg[i-1][j])+
                int(noiseImg[i-1][j+1])+
                int(noiseImg[i][j-1])+
                int(noiseImg[i][j])+
                int(noiseImg[i][j+1])+
                int(noiseImg[i+1][j-1])+
                int(noiseImg[i+1][j])+
                int(noiseImg[i+1][j+1])
            )
            #calculating average
            average=total//9
            filterImg[i][j]=average
    boxImg=np.zeros((rows, columns), dtype=np.uint8)
    #box filter kernel
    kernel=np.array([[1, 1, 1,], [1, 1, 1,], [1, 1, 1,]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(noiseImg[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            boxImg[i][j]=totalSum//9

    gaussianImg=np.zeros((rows, columns), dtype=np.uint8)
    #box filter kernel
    kernel=np.array([[1, 2, 1,], [2, 4, 2,], [1, 2, 1,]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(noiseImg[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing by dividing with sum of kernel values (16)
            gaussianImg[i][j]=totalSum//16

    medianImg=np.zeros((rows, columns), dtype=np.uint8)
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            pixels=[]#storing neighboring pixels
            #collecting all 9 neighboring pixels
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(noiseImg[i+m-1][j+n-1])#getting corresponding image pixel
                    pixels.append(pixel)
            #bubble sorting
            for x in range(len(pixels)-1):
                for y in range(len(pixels)-x-1):
                    if pixels[y]>pixels[y+1]:
                        temp=pixels[y]
                        pixels[y]=pixels[y+1]
                        pixels[y+1]=temp
            median=pixels[4]
            medianImg[i][j]=median
    
    return noiseImg, filterImg, boxImg, gaussianImg, medianImg

def kernel_comparison(img):
    rows, columns=img.shape
    filterImg3=np.zeros((rows, columns), dtype=np.uint8)
    filterImg5=np.zeros((rows, columns), dtype=np.uint8)
    filterImg7=np.zeros((rows, columns), dtype=np.uint8)
    #for 3x3 kernel
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            #calculating sum of all 9 neighboring pixels
            total=0
            for m in range(3):
                for n in range(3):
                    total=total+int(img[i+m-1][j+n-1])
            #calculating average
            average=total//9
            filterImg3[i][j]=average
            
    #for 5x5 kernel
    #loop through all pixels except border pixels
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            #calculating sum of all 9 neighboring pixels
            total=0
            for m in range(5):
                for n in range(5):
                    total=total+int(img[i+m-2][j+n-2])
            #calculating average
            average=total//25
            filterImg5[i][j]=average
            
    #for 7x7 kernel
    #loop through all pixels except border pixels
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            #calculating sum of all 9 neighboring pixels
            total=0
            for m in range(7):
                for n in range(7):
                    total=total+int(img[i+m-3][j+n-3])
            #calculating average
            average=total//49
            filterImg7[i][j]=average
    
    #box filter
    boxImg3=np.zeros((rows, columns), dtype=np.uint8)
    boxImg5=np.zeros((rows, columns), dtype=np.uint8)
    boxImg7=np.zeros((rows, columns), dtype=np.uint8)
    
    #for 3x3 kernel
    kernel3=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel3[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            boxImg3[i][j]=totalSum//9
            
    #for 5x5 kernel
    kernel5=np.array([[1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1]])
    #loop through all pixels except border pixels
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            totalSum=0
            for m in range(5):#through rows
                for n in range(5):#through columns
                    pixel=int(img[i+m-2][j+n-2])#getting corresponding image pixel
                    weight=kernel5[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            boxImg5[i][j]=totalSum//25
            
    #for 7x7 kernel
    kernel7=np.array([[1, 1, 1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1]])
    #loop through all pixels except border pixels
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            totalSum=0
            for m in range(7):#through rows
                for n in range(7):#through columns
                    pixel=int(img[i+m-3][j+n-3])#getting corresponding image pixel
                    weight=kernel7[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            boxImg7[i][j]=totalSum//49
    
    #gaussian filter
    gaussianImg3=np.zeros((rows, columns), dtype=np.uint8)
    gaussianImg5=np.zeros((rows, columns), dtype=np.uint8)
    gaussianImg7=np.zeros((rows, columns), dtype=np.uint8)
    
    #for 3x3 kernel
    kernel3=np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel3[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            gaussianImg3[i][j]=totalSum//16
            
    #for 5x5 kernel
    kernel5=np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
    #loop through all pixels except border pixels
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            totalSum=0
            for m in range(5):#through rows
                for n in range(5):#through columns
                    pixel=int(img[i+m-2][j+n-2])#getting corresponding image pixel
                    weight=kernel5[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            gaussianImg5[i][j]=totalSum//256
            
    #for 7x7 kernel
    kernel7=np.array([[0, 0, 1, 2, 1, 0, 0], 
                        [0, 3, 13, 22, 13, 3, 0], 
                        [1, 13, 59, 97, 59, 13, 1],
                        [2, 22, 97, 159, 97, 22, 2],
                        [1, 13, 59, 97, 59, 13, 1],
                        [0, 3, 13, 22, 13, 3, 0],
                        [0, 0, 1, 2, 1, 0, 0]])
    #loop through all pixels except border pixels
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            totalSum=0
            for m in range(7):#through rows
                for n in range(7):#through columns
                    pixel=int(img[i+m-3][j+n-3])#getting corresponding image pixel
                    weight=kernel7[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            gaussianImg7[i][j]=totalSum//1003
    
    #median filter
    medianImg3=np.zeros((rows, columns), dtype=np.uint8)
    medianImg5=np.zeros((rows, columns), dtype=np.uint8)
    medianImg7=np.zeros((rows, columns), dtype=np.uint8)
    
    #for 3x3 
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            pixels=[]#storing neighboring pixels
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    pixels.append(pixel)
            #bubble sorting
            for x in range(len(pixels)-1):
                for y in range(len(pixels)-x-1):
                    if pixels[y]>pixels[y+1]:
                        temp=pixels[y]
                        pixels[y]=pixels[y+1]
                        pixels[y+1]=temp
            #normalizing
            medianImg3[i][j]=pixels[4]
            
    #for 5x5 kernel
    #loop through all pixels except border pixels
    for i in range(2, rows-2):
        for j in range(2, columns-2):
            pixels=[]#storing neighboring pixels
            for m in range(5):#through rows
                for n in range(5):#through columns
                    pixel=int(img[i+m-2][j+n-2])#getting corresponding image pixel
                    pixels.append(pixel)
            #bubble sorting
            for x in range(len(pixels)-1):
                for y in range(len(pixels)-x-1):
                    if pixels[y]>pixels[y+1]:
                        temp=pixels[y]
                        pixels[y]=pixels[y+1]
                        pixels[y+1]=temp
            medianImg5[i][j]=pixels[12]
            
    #for 7x7 kernel
    #loop through all pixels except border pixels
    for i in range(3, rows-3):
        for j in range(3, columns-3):
            pixels=[]#storing neighboring pixels
            for m in range(7):#through rows
                for n in range(7):#through columns
                    pixel=int(img[i+m-3][j+n-3])#getting corresponding image pixel
                    pixels.append(pixel)
            #bubble sorting
            for x in range(len(pixels)-1):
                for y in range(len(pixels)-x-1):
                    if pixels[y]>pixels[y+1]:
                        temp=pixels[y]
                        pixels[y]=pixels[y+1]
                        pixels[y+1]=temp
            medianImg7[i][j]=pixels[24]
    return (filterImg3, filterImg5, filterImg7,
            boxImg3, boxImg5, boxImg7,
            gaussianImg3, gaussianImg5, gaussianImg7,
            medianImg3, medianImg5, medianImg7)

def execution_time(img):
    rows, columns=img.shape
    meanStart=time.perf_counter()
    filterImg3=np.zeros((rows, columns), dtype=np.uint8)
    #for 3x3 kernel
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            #calculating sum of all 9 neighboring pixels
            total=0
            for m in range(3):
                for n in range(3):
                    total=total+int(img[i+m-1][j+n-1])
            #calculating average
            average=total//9
            filterImg3[i][j]=average
    meanEnd=time.perf_counter()
    meanTime=meanEnd-meanStart

    #box filter
    boxStart=time.perf_counter()
    boxImg3=np.zeros((rows, columns), dtype=np.uint8)
    #for 3x3 kernel
    kernel3=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel3[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            boxImg3[i][j]=totalSum//9
    boxEnd=time.perf_counter()
    boxTime=boxEnd-boxStart
    
    #gaussian filter
    gaussianStart=time.perf_counter()
    gaussianImg3=np.zeros((rows, columns), dtype=np.uint8)
    #for 3x3 kernel
    kernel3=np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            totalSum=0
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    weight=kernel3[m][n]#getting corresponding kernel value
                    totalSum=totalSum+(pixel*weight)#multiply and add
            #normalizing
            gaussianImg3[i][j]=totalSum//16
    gaussianEnd=time.perf_counter()
    gaussianTime=gaussianEnd-gaussianStart
    
    #median filter
    medianStart=time.perf_counter()
    medianImg3=np.zeros((rows, columns), dtype=np.uint8)
    #for 3x3 
    #loop through all pixels except border pixels
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            pixels=[]#storing neighboring pixels
            for m in range(3):#through rows
                for n in range(3):#through columns
                    pixel=int(img[i+m-1][j+n-1])#getting corresponding image pixel
                    pixels.append(pixel)
            #bubble sorting
            for x in range(len(pixels)-1):
                for y in range(len(pixels)-x-1):
                    if pixels[y]>pixels[y+1]:
                        temp=pixels[y]
                        pixels[y]=pixels[y+1]
                        pixels[y+1]=temp
            #normalizing
            medianImg3[i][j]=pixels[4]
    medianEnd=time.perf_counter()
    medianTime=medianEnd-medianStart
    return (f"Mean Filter: {meanTime:.6f} seconds\n"
            f"Box Filter: {boxTime:.6f} seconds\n"
            f"Gaussian Filter: {gaussianTime:.6f} seconds\n"
            f"Median Filter: {medianTime:.6f} seconds\n")

def apply_filters(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Task 1
    meanImg = mean_filter(gray)
    boxImg = box_filter(gray)
    gaussianImg = gaussian_filter(gray)
    medianImg = median_filter(gray)

    # Task 5
    noisyImg, compareMean, compareBox, compareGaussian, compareMedian = comparison_filters(gray)

    # Task 6
    (mean3, mean5, mean7,
     box3, box5, box7,
     gaussian3, gaussian5, gaussian7,
     median3, median5, median7) = kernel_comparison(gray)

    # Task 7 (Execution Time Only)
    executionText = execution_time(gray)

    return (
        gray,
        meanImg,
        boxImg,
        gaussianImg,
        medianImg,

        noisyImg,
        compareMean,
        compareBox,
        compareGaussian,
        compareMedian,

        mean3,
        mean5,
        mean7,

        box3,
        box5,
        box7,

        gaussian3,
        gaussian5,
        gaussian7,

        median3,
        median5,
        median7,

        executionText
    )

outputs=[
    gr.Image(label="Original Image"),

    gr.Image(label="Mean Filter"),
    gr.Image(label="Box Filter"),
    gr.Image(label="Gaussian Filter"),
    gr.Image(label="Median Filter"),

    gr.Image(label="Noisy Image"),
    gr.Image(label="Mean Comparison"),
    gr.Image(label="Box Comparison"),
    gr.Image(label="Gaussian Comparison"),
    gr.Image(label="Median Comparison"),

    gr.Image(label="Mean 3x3"),
    gr.Image(label="Mean 5x5"),
    gr.Image(label="Mean 7x7"),

    gr.Image(label="Box 3x3"),
    gr.Image(label="Box 5x5"),
    gr.Image(label="Box 7x7"),

    gr.Image(label="Gaussian 3x3"),
    gr.Image(label="Gaussian 5x5"),
    gr.Image(label="Gaussian 7x7"),

    gr.Image(label="Median 3x3"),
    gr.Image(label="Median 5x5"),
    gr.Image(label="Median 7x7"),

    gr.Textbox(label="Execution Time")
]


with gr.Blocks(title="Image Filtering (Manual Implementation)") as demo:

    gr.Markdown("# Manual Image Filtering")
    gr.Markdown("Upload an image to test all tasks of Module 5.")

    inputImage = gr.Image(label="Upload Image")

    with gr.Tab("Task 1,2,3,4 - Filters"):

        run1 = gr.Button("Apply Filters")

        original = gr.Image(label="Original Image")

        mean = gr.Image(label="Mean Filter")
        box = gr.Image(label="Box Filter")
        gaussian = gr.Image(label="Gaussian Filter")
        median = gr.Image(label="Median Filter")

    with gr.Tab("Task 5 - Noise Comparison"):

        run2 = gr.Button("Compare Filters")

        noisy = gr.Image(label="Noisy Image")

        meanCompare = gr.Image(label="Mean Filter")
        boxCompare = gr.Image(label="Box Filter")
        gaussianCompare = gr.Image(label="Gaussian Filter")
        medianCompare = gr.Image(label="Median Filter")

    with gr.Tab("Task 6 - Kernel Comparison"):

        run3 = gr.Button("Compare Kernels")

        mean3 = gr.Image(label="Mean 3x3")
        mean5 = gr.Image(label="Mean 5x5")
        mean7 = gr.Image(label="Mean 7x7")

        box3 = gr.Image(label="Box 3x3")
        box5 = gr.Image(label="Box 5x5")
        box7 = gr.Image(label="Box 7x7")

        gaussian3 = gr.Image(label="Gaussian 3x3")
        gaussian5 = gr.Image(label="Gaussian 5x5")
        gaussian7 = gr.Image(label="Gaussian 7x7")

        median3 = gr.Image(label="Median 3x3")
        median5 = gr.Image(label="Median 5x5")
        median7 = gr.Image(label="Median 7x7")

    with gr.Tab("Task 7 - Execution Time"):

        run4 = gr.Button("Measure Time")

        execution = gr.Textbox(label="Execution Time")

    run1.click(
        fn=lambda img: (
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
            mean_filter(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)),
            box_filter(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)),
            gaussian_filter(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)),
            median_filter(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        ),
        inputs=inputImage,
        outputs=[original, mean, box, gaussian, median]
    )

    run2.click(
        fn=lambda img: comparison_filters(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)),
        inputs=inputImage,
        outputs=[
            noisy,
            meanCompare,
            boxCompare,
            gaussianCompare,
            medianCompare
        ]
    )

    run3.click(
        fn=lambda img: kernel_comparison(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)),
        inputs=inputImage,
        outputs=[
            mean3, mean5, mean7,
            box3, box5, box7,
            gaussian3, gaussian5, gaussian7,
            median3, median5, median7
        ]
    )

    run4.click(
        fn=lambda img: execution_time(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)),
        inputs=inputImage,
        outputs=execution
    )

demo.launch()