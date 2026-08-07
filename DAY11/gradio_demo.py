import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def rgbToGray(image):
    rows, columns, channel=image.shape
    gray=np.zeros((rows, columns),dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=image[i][j][0]
            green=image[i][j][1]
            red=image[i][j][2]
            grayVal=int(0.114*blue+0.587*green+0.299*red)
            gray[i][j]=grayVal
    return gray

def bgrToRgb(image):
    rows, columns, channel=image.shape
    rgbImg=np.zeros((rows, columns, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(columns):
            blue=image[i][j][0]
            green=image[i][j][1]
            red=image[i][j][2]

            rgbImg[i][j][0]=red
            rgbImg[i][j][1]=green
            rgbImg[i][j][2]=blue

    return rgbImg

def channelSep(image):
    rows, columns, channel=image.shape
    blue=np.zeros((rows, columns), dtype=np.uint8)
    green=np.zeros((rows, columns), dtype=np.uint8)
    red=np.zeros((rows, columns), dtype=np.uint8)

    for i in range(rows):
        for j in range(columns):
            blue[i][j]=image[i][j][0]
            green[i][j]=image[i][j][1]
            red[i][j]=image[i][j][2]
    return blue, green, red

def histogram(image):
    hisArr=np.zeros(256, dtype=int)
    rows, columns=image.shape
    for i in range(rows):
        for j in range(columns):
            pixel=image[i, j]
            hisArr[pixel]+=1
    return hisArr

def cdfCalculation(histogram):
    cdf=np.zeros(256, dtype=int)
    cdf[0]=histogram[0]
    for i in range(1, 256):
        cdf[i]=cdf[i-1]+histogram[i]
    return cdf

def normalizeCdf(histogram):
    cdf=np.zeros(256, dtype=float)
    cdf[0]=histogram[0]
    for i in range(1, 256):
        cdf[i]=cdf[i-1]+histogram[i]
    cdf=cdf/cdf[-1]
    return cdf

def histEqualization(image):
    histImg=histogram(image)
    cdf=np.zeros(256, dtype=int)
    cdf[0]=histImg[0]
    for i in range(1, 256):
        cdf[i]=cdf[i-1]+histImg[i]

    cdfMin=0
    for value in cdf:
        if value!=0:
            cdfMin=value
            break
    pixelTotal=image.shape[0]*image.shape[1]

    lookup=np.zeros(256, dtype=np.uint8)
    for i in range(256):
        value=((cdf[i]-cdfMin)/(pixelTotal-cdfMin))*255
        if value<0:
            value=0
        lookup[i]=int(value)

    rows, columns=image.shape
    equalizedImg=np.zeros_like(image)
    for i in range(rows):
        for j in range(columns):
            pixel=image[i][j]
            equalizedImg[i][j]=lookup[pixel]
    return equalizedImg

def clipHist(histogram, clipLimit):
    histClipped=histogram.copy()
    extraPix=0
    for i in range(256):
        if histClipped[i]>clipLimit:
            extraPix+=histClipped[i]-clipLimit
            histClipped[i]=clipLimit
    return histClipped, extraPix

def pixelsRedistribution(histogram, extraPix):
    value=extraPix//256
    remainder=extraPix%256
    for i in range(256):
        histogram[i]+=value

    for i in range(remainder):
        histogram[i]+=1
    return histogram

def tileEqualize(tile, clipLimit):
    hist=histogram(tile)
    histClipped, extraPix=clipHist(hist, clipLimit)
    histClipped=pixelsRedistribution(histClipped, extraPix)
    cdf=cdfCalculation(histClipped)
    cdfMin=0
    for value in cdf:
        if value!=0:
            cdfMin=value
            break
    totalPix=tile.shape[0]*tile.shape[1]
    lookupTable=np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if totalPix==cdfMin:
            lookupTable[i]=i
        else:
            newVal=((cdf[i]-cdfMin)/(totalPix-cdfMin))*255
            if newVal<0:
                newVal=0
            if newVal>255:
                newVal=255
            lookupTable[i]=int(newVal)

    rows, columns=tile.shape
    output=np.zeros_like(tile)
    for i in range(rows):
        for j in range(columns):
            pixel=tile[i][j]
            output[i][j]=lookupTable[pixel]
    return output

def clahe(image, tileSize=8, clipLimit=40):
    rows, columns=image.shape
    output=np.zeros_like(image)
    for i in range(0, rows, tileSize):
        for j in range(0, columns, tileSize):
            rowEnd=i+tileSize
            colEnd=j+tileSize
            if rowEnd>rows:
                rowEnd=rows
            if colEnd>columns:
                colEnd=columns
            tileRow=rowEnd-i
            tileCol=colEnd-j
            tile=np.zeros((tileRow, tileCol), dtype=np.uint8)
            for m in range(tileRow):
                for n in range(tileCol):
                    tile[m][n]=image[i+m][j+n]
            equaTile=tileEqualize(tile, clipLimit)
            for r in range(tileRow):
                for c in range(tileCol):
                    output[i+r][j+c]=equaTile[r][c]
    return output

def mapping(source, reference):
    map=np.zeros(256, dtype=np.uint8)
    for i in range(256):
        difference=abs(source[i]-reference[0])
        index=0
        for j in range(256):
            d1=abs(source[i]-reference[j])
            if d1<difference:
                difference=d1
                index=j

        map[i]=index
    return map

def histogramMatch(srcImg, refImg):
    srcHist=histogram(srcImg)
    refHist=histogram(refImg)

    srcCdf=normalizeCdf(srcHist)
    refCdf=normalizeCdf(refHist)

    map=mapping(srcCdf, refCdf)
    rows, columns=srcImg.shape
    match=np.zeros_like(srcImg)
    for i in range(rows):
        for j in range(columns):
            pixel=srcImg[i][j]
            match[i][j]=map[pixel]
    return match

def plotHistogram(hist, title):
    figure=plt.figure(figsize=(5, 3))
    plt.plot(range(256), hist)
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.grid(True)
    return figure

def numpyToPIL(image):
    return Image.fromarray(image)

def pilToNumpy(image):
    return np.array(image)

def plotRGBHistogram(redHist, greenHist, blueHist, title):

    figure = plt.figure(figsize=(5,3))

    plt.plot(range(256), redHist, color="red", label="Red")
    plt.plot(range(256), greenHist, color="green", label="Green")
    plt.plot(range(256), blueHist, color="blue", label="Blue")

    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.legend()

    return figure

def task1(image1, image2):

    image1 = pilToNumpy(image1)
    image2 = pilToNumpy(image2)

    gray1 = rgbToGray(image1)
    gray2 = rgbToGray(image2)

    hist1 = histogram(gray1)
    hist2 = histogram(gray2)

    fig1 = plotHistogram(hist1, "Histogram: Image 1")
    fig2 = plotHistogram(hist2, "Histogram: Image 2")

    observation = (
        "Dark images have most histogram values on left side.\n"
        "Bright images have most histogram values on right side.\n"
        "Low contrast images have narrow histogram.\n"
        "High contrast images have wider histogram."
    )

    return (
        numpyToPIL(gray1),
        fig1,

        numpyToPIL(gray2),
        fig2,

        observation
    )

def task2(image1, image2):

    image1 = pilToNumpy(image1)
    image2 = pilToNumpy(image2)

    gray1 = rgbToGray(image1)
    gray2 = rgbToGray(image2)

    hist1 = histogram(gray1)
    hist2 = histogram(gray2)

    fig1 = plotHistogram(hist1, "Histogram: Image 1")
    fig2 = plotHistogram(hist2, "Histogram: Image 2")

    histogramValues1 = "Histogram Values of Image 1\n"
    histogramValues1 += "Intensity\tFrequency\n"
    histogramValues1 += "-" * 25 + "\n"

    for i in range(256):
        histogramValues1 += f"{i}\t\t{hist1[i]}\n"

    histogramValues2 = "Histogram Values of Image 2\n"
    histogramValues2 += "Intensity\tFrequency\n"
    histogramValues2 += "-" * 25 + "\n"

    for i in range(256):
        histogramValues2 += f"{i}\t\t{hist2[i]}\n"

    return (
        numpyToPIL(gray1),
        fig1,
        histogramValues1,

        numpyToPIL(gray2),
        fig2,
        histogramValues2
    )

def task3(image1, image2):

    image1 = pilToNumpy(image1)
    image2 = pilToNumpy(image2)

    rgbImage1 = image1
    rgbImage2 = image2

    red1, green1, blue1 = channelSep(rgbImage1)
    red2, green2, blue2 = channelSep(rgbImage2)

    redHist1 = histogram(red1)
    greenHist1 = histogram(green1)
    blueHist1 = histogram(blue1)

    redHist2 = histogram(red2)
    greenHist2 = histogram(green2)
    blueHist2 = histogram(blue2)

    figure1 = plotRGBHistogram(
        redHist1,
        greenHist1,
        blueHist1,
        "RGB Histogram Image 1"
    )

    figure2 = plotRGBHistogram(
        redHist2,
        greenHist2,
        blueHist2,
        "RGB Histogram Image 2"
    )

    observation = (
        "Each RGB channel has its own histogram.\n"
        "Red histogram shows distribution of red pixels.\n"
        "Green histogram shows distribution of green pixels.\n"
        "Blue histogram shows distribution of blue pixels.\n"
        "Different images produce different RGB histogram patterns.\n"
        "Dominant color channels have higher frequency values."
    )

    return (
        numpyToPIL(rgbImage1),
        figure1,

        numpyToPIL(rgbImage2),
        figure2,

        observation
    )

def task4(image):

    image = pilToNumpy(image)

    grayImage = rgbToGray(image)

    equalizedImage = histEqualization(grayImage)

    originalHistogram = histogram(grayImage)
    equalizedHistogram = histogram(equalizedImage)

    figure1 = plotHistogram(
        originalHistogram,
        "Original Histogram"
    )

    figure2 = plotHistogram(
        equalizedHistogram,
        "Equalized Histogram"
    )

    observation = (
        "Histogram Equalization improves image contrast.\n"
        "Pixel intensities spread over a wider range.\n"
        "Dark regions become more visible.\n"
        "The equalized histogram is more uniformly distributed."
    )

    return (
        numpyToPIL(grayImage),
        numpyToPIL(equalizedImage),
        figure1,
        figure2,
        observation
    )

def task5(image1, image2):

    image1 = pilToNumpy(image1)
    image2 = pilToNumpy(image2)

    gray1 = rgbToGray(image1)
    gray2 = rgbToGray(image2)

    equalize1 = histEqualization(gray1)
    equalize2 = histEqualization(gray2)

    clahe1 = clahe(gray1, tileSize=8, clipLimit=40)
    clahe2 = clahe(gray2, tileSize=8, clipLimit=40)

    ogHist1 = histogram(gray1)
    eqHist1 = histogram(equalize1)
    claheHist1 = histogram(clahe1)

    ogHist2 = histogram(gray2)
    eqHist2 = histogram(equalize2)
    claheHist2 = histogram(clahe2)

    figOriginal1 = plotHistogram(ogHist1, "Original Histogram")
    figEqualized1 = plotHistogram(eqHist1, "Equalized Histogram")
    figClahe1 = plotHistogram(claheHist1, "CLAHE Histogram")

    figOriginal2 = plotHistogram(ogHist2, "Original Histogram")
    figEqualized2 = plotHistogram(eqHist2, "Equalized Histogram")
    figClahe2 = plotHistogram(claheHist2, "CLAHE Histogram")

    observation = (
        "Histogram Equalization improves overall image contrast.\n"
        "CLAHE enhances local contrast by processing small image tiles.\n"
        "CLAHE preserves local details better than Histogram Equalization.\n"
        "Histogram Equalization may over-enhance some regions.\n"
        "CLAHE usually produces a more natural-looking image."
    )

    return (

        numpyToPIL(gray1),
        numpyToPIL(equalize1),
        numpyToPIL(clahe1),

        figOriginal1,
        figEqualized1,
        figClahe1,

        numpyToPIL(gray2),
        numpyToPIL(equalize2),
        numpyToPIL(clahe2),

        figOriginal2,
        figEqualized2,
        figClahe2,

        observation
    )

def task6(sourceImage, referenceImage):

    sourceImage = pilToNumpy(sourceImage)
    referenceImage = pilToNumpy(referenceImage)

    srcGray = rgbToGray(sourceImage)
    refGray = rgbToGray(referenceImage)

    matchedImage = histogramMatch(srcGray, refGray)

    srcHist = histogram(srcGray)
    refHist = histogram(refGray)
    matchedHist = histogram(matchedImage)

    srcFig = plotHistogram(srcHist, "Source Histogram")
    refFig = plotHistogram(refHist, "Reference Histogram")
    matchedFig = plotHistogram(matchedHist, "Matched Histogram")

    observation = (
        "Histogram Matching changes the intensity distribution of the source image.\n"
        "The matched image becomes similar to the reference image.\n"
        "The matched histogram follows the shape of the reference histogram.\n"
        "Unlike Histogram Equalization, Histogram Matching uses another image as a reference."
    )

    return (

        numpyToPIL(srcGray),
        srcFig,

        numpyToPIL(refGray),
        refFig,

        numpyToPIL(matchedImage),
        matchedFig,

        observation
    )

def task7(image1, image2, reference):

    image1 = pilToNumpy(image1)
    image2 = pilToNumpy(image2)
    reference = pilToNumpy(reference)

    gray1 = rgbToGray(image1)
    gray2 = rgbToGray(image2)
    refGray = rgbToGray(reference)

    equalize1 = histEqualization(gray1)
    equalize2 = histEqualization(gray2)

    clahe1 = clahe(gray1, tileSize=8, clipLimit=40)
    clahe2 = clahe(gray2, tileSize=8, clipLimit=40)

    match1 = histogramMatch(gray1, refGray)
    match2 = histogramMatch(gray2, refGray)

    figOg1 = plotHistogram(histogram(gray1), "Original Histogram")
    figEq1 = plotHistogram(histogram(equalize1), "Histogram Equalization")
    figClahe1 = plotHistogram(histogram(clahe1), "CLAHE")
    figMatch1 = plotHistogram(histogram(match1), "Histogram Matching")

    figOg2 = plotHistogram(histogram(gray2), "Original Histogram")
    figEq2 = plotHistogram(histogram(equalize2), "Histogram Equalization")
    figClahe2 = plotHistogram(histogram(clahe2), "CLAHE")
    figMatch2 = plotHistogram(histogram(match2), "Histogram Matching")

    observation = (
        "Histogram Equalization improves overall image contrast.\n"
        "Histogram Matching changes the source image according to the reference image.\n"
        "Histogram Equalization is suitable for general contrast enhancement.\n"
        "Histogram Matching is useful when one image should resemble another.\n"
        "CLAHE improves local contrast by processing small image tiles.\n"
        "CLAHE preserves image details better than Histogram Equalization.\n"
        "CLAHE is more suitable for low contrast images because it enhances local details."
    )

    return (

        numpyToPIL(gray1),
        numpyToPIL(equalize1),
        numpyToPIL(clahe1),
        numpyToPIL(match1),

        figOg1,
        figEq1,
        figClahe1,
        figMatch1,

        numpyToPIL(gray2),
        numpyToPIL(equalize2),
        numpyToPIL(clahe2),
        numpyToPIL(match2),

        figOg2,
        figEq2,
        figClahe2,
        figMatch2,

        observation
    )

with gr.Blocks(title="Image Histogram Processing") as demo:

    gr.Markdown(
        """
        # Image Histogram Processing (Manual Implementation)

        Upload the required images for each task.
        """
    )

    with gr.Tabs():

        with gr.Tab("Task 1"):

            gr.Markdown("## Task 1 - Histogram Basics")

            with gr.Row():

                inputImage1 = gr.Image(
                    type="pil",
                    label="Upload Grayscale Image 1 (e.g. cat.jpeg)"
                )

                inputImage2 = gr.Image(
                    type="pil",
                    label="Upload Grayscale Image 2 (e.g. lowCon.jpg)"
                )

            runBtn = gr.Button("Generate Histogram")

            with gr.Row():

                outputImage1 = gr.Image(label="Grayscale Image 1")
                outputHistogram1 = gr.Plot(label="Histogram 1")

            with gr.Row():

                outputImage2 = gr.Image(label="Grayscale Image 2")
                outputHistogram2 = gr.Plot(label="Histogram 2")

            observation = gr.Textbox(
                label="Observation",
                lines=4
            )

            runBtn.click(
                fn=task1,
                inputs=[
                    inputImage1,
                    inputImage2
                ],
                outputs=[
                    outputImage1,
                    outputHistogram1,

                    outputImage2,
                    outputHistogram2,

                    observation
                ]
            )

        with gr.Tab("Task 2"):

            gr.Markdown("## Task 2 - Grayscale Histogram")

            with gr.Row():

                inputImage1 = gr.Image(
                    type="pil",
                    label="Upload Grayscale Image 1 (e.g. cat.jpeg)"
                )

                inputImage2 = gr.Image(
                    type="pil",
                    label="Upload Grayscale Image 2 (e.g. lowCon.jpg)"
                )

            runBtn = gr.Button("Generate Histogram")

            with gr.Row():

                outputImage1 = gr.Image(label="Grayscale Image 1")
                outputHistogram1 = gr.Plot(label="Histogram 1")

            histogramText1 = gr.Textbox(
                label="Histogram Values (Image 1)",
                lines=12
            )

            with gr.Row():

                outputImage2 = gr.Image(label="Grayscale Image 2")
                outputHistogram2 = gr.Plot(label="Histogram 2")

            histogramText2 = gr.Textbox(
                label="Histogram Values (Image 2)",
                lines=12
            )

            runBtn.click(
                fn=task2,
                inputs=[
                    inputImage1,
                    inputImage2
                ],
                outputs=[
                    outputImage1,
                    outputHistogram1,
                    histogramText1,

                    outputImage2,
                    outputHistogram2,
                    histogramText2
                ]
            )

        with gr.Tab("Task 3"):

            gr.Markdown("## Task 3 - RGB Histogram")

            with gr.Row():

                inputImage1 = gr.Image(
                    type="pil",
                    label="Upload RGB Image 1 (e.g. dog.jpg)"
                )

                inputImage2 = gr.Image(
                    type="pil",
                    label="Upload RGB Image 2 (e.g. jpeg_43-2.jpg)"
                )

            runBtn = gr.Button("Generate RGB Histogram")

            with gr.Row():

                outputImage1 = gr.Image(label="RGB Image 1")
                outputHistogram1 = gr.Plot(label="RGB Histogram 1")

            with gr.Row():

                outputImage2 = gr.Image(label="RGB Image 2")
                outputHistogram2 = gr.Plot(label="RGB Histogram 2")

            observation = gr.Textbox(
                label="Observation",
                lines=6
            )

            runBtn.click(
                fn=task3,
                inputs=[
                    inputImage1,
                    inputImage2
                ],
                outputs=[
                    outputImage1,
                    outputHistogram1,

                    outputImage2,
                    outputHistogram2,

                    observation
                ]
            )

        with gr.Tab("Task 4"):

            gr.Markdown("## Task 4 - Histogram Equalization")

            inputImage = gr.Image(
                type="pil",
                label="Upload RGB Image (e.g. dog.jpg)"
            )

            runBtn = gr.Button("Apply Histogram Equalization")

            with gr.Row():

                outputImage1 = gr.Image(
                    label="Original Grayscale Image"
                )

                outputImage2 = gr.Image(
                    label="Equalized Image"
                )

            with gr.Row():

                outputHistogram1 = gr.Plot(
                    label="Original Histogram"
                )

                outputHistogram2 = gr.Plot(
                    label="Equalized Histogram"
                )

            observation = gr.Textbox(
                label="Observation",
                lines=4
            )

            runBtn.click(
                fn=task4,
                inputs=inputImage,
                outputs=[
                    outputImage1,
                    outputImage2,
                    outputHistogram1,
                    outputHistogram2,
                    observation
                ]
            )

        with gr.Tab("Task 5"):

            gr.Markdown("## Task 5 - CLAHE")

            with gr.Row():

                inputImage1 = gr.Image(
                    type="pil",
                    label="Upload Low Contrast Image 1 (e.g. lowCon.jpg)"
                )

                inputImage2 = gr.Image(
                    type="pil",
                    label="Upload Low Contrast Image 2 (e.g. lowImg2.jpeg)"
                )

            runBtn = gr.Button("Compare Histogram Equalization and CLAHE")

            with gr.Tabs():

                # ---------------- IMAGE 1 ---------------- #

                with gr.Tab("Image 1"):

                    with gr.Row():

                        outputGray1 = gr.Image(label="Original Image")
                        outputEqualized1 = gr.Image(label="Histogram Equalization")
                        outputClahe1 = gr.Image(label="CLAHE")

                    with gr.Row():

                        outputHistOriginal1 = gr.Plot(label="Original Histogram")
                        outputHistEqualized1 = gr.Plot(label="Equalized Histogram")
                        outputHistClahe1 = gr.Plot(label="CLAHE Histogram")

                # ---------------- IMAGE 2 ---------------- #

                with gr.Tab("Image 2"):

                    with gr.Row():

                        outputGray2 = gr.Image(label="Original Image")
                        outputEqualized2 = gr.Image(label="Histogram Equalization")
                        outputClahe2 = gr.Image(label="CLAHE")

                    with gr.Row():

                        outputHistOriginal2 = gr.Plot(label="Original Histogram")
                        outputHistEqualized2 = gr.Plot(label="Equalized Histogram")
                        outputHistClahe2 = gr.Plot(label="CLAHE Histogram")

            observation = gr.Textbox(
                label="Observation",
                lines=6
            )

            runBtn.click(
                fn=task5,
                inputs=[
                    inputImage1,
                    inputImage2
                ],
                outputs=[
                    outputGray1,
                    outputEqualized1,
                    outputClahe1,

                    outputHistOriginal1,
                    outputHistEqualized1,
                    outputHistClahe1,

                    outputGray2,
                    outputEqualized2,
                    outputClahe2,

                    outputHistOriginal2,
                    outputHistEqualized2,
                    outputHistClahe2,

                    observation
                ]
            )

        with gr.Tab("Task 6"):

            gr.Markdown("## Task 6 - Histogram Matching")

            with gr.Row():

                sourceImage = gr.Image(
                    type="pil",
                    label="Upload Source Image (e.g. lowCon.jpg)"
                )

                referenceImage = gr.Image(
                    type="pil",
                    label="Upload Reference Image (e.g. cat.jpeg)"
                )

            runBtn = gr.Button("Apply Histogram Matching")

            with gr.Tabs():

                with gr.Tab("Source Image"):

                    outputSource = gr.Image(label="Source Image")
                    outputSourceHist = gr.Plot(label="Source Histogram")

                with gr.Tab("Reference Image"):

                    outputReference = gr.Image(label="Reference Image")
                    outputReferenceHist = gr.Plot(label="Reference Histogram")

                with gr.Tab("Matched Image"):

                    outputMatched = gr.Image(label="Matched Image")
                    outputMatchedHist = gr.Plot(label="Matched Histogram")

            observation = gr.Textbox(
                label="Observation",
                lines=5
            )

            runBtn.click(
                fn=task6,
                inputs=[
                    sourceImage,
                    referenceImage
                ],
                outputs=[
                    outputSource,
                    outputSourceHist,

                    outputReference,
                    outputReferenceHist,

                    outputMatched,
                    outputMatchedHist,

                    observation
                ]
            )

        with gr.Tab("Task 7"):

            gr.Markdown("## Task 7 - Final Comparison")

            with gr.Row():

                inputImage1 = gr.Image(
                    type="pil",
                    label="Upload Low Contrast Image 1"
                )

                inputImage2 = gr.Image(
                    type="pil",
                    label="Upload Low Contrast Image 2"
                )

                referenceImage = gr.Image(
                    type="pil",
                    label="Upload Reference Image (e.g. cat.jpeg)"
                )

            runBtn = gr.Button("Compare All Methods")

            with gr.Tabs():

                # ---------------- IMAGE 1 ---------------- #

                with gr.Tab("Image 1"):

                    with gr.Tabs():

                        with gr.Tab("Images"):

                            with gr.Row():

                                outputGray1 = gr.Image(label="Original")
                                outputEqualized1 = gr.Image(label="Histogram Equalization")
                                outputClahe1 = gr.Image(label="CLAHE")
                                outputMatch1 = gr.Image(label="Histogram Matching")

                        with gr.Tab("Histograms"):

                            with gr.Row():

                                outputHistGray1 = gr.Plot(label="Original Histogram")
                                outputHistEq1 = gr.Plot(label="Equalized Histogram")
                                outputHistClahe1 = gr.Plot(label="CLAHE Histogram")
                                outputHistMatch1 = gr.Plot(label="Matched Histogram")

                # ---------------- IMAGE 2 ---------------- #

                with gr.Tab("Image 2"):

                    with gr.Tabs():

                        with gr.Tab("Images"):

                            with gr.Row():

                                outputGray2 = gr.Image(label="Original")
                                outputEqualized2 = gr.Image(label="Histogram Equalization")
                                outputClahe2 = gr.Image(label="CLAHE")
                                outputMatch2 = gr.Image(label="Histogram Matching")

                        with gr.Tab("Histograms"):

                            with gr.Row():

                                outputHistGray2 = gr.Plot(label="Original Histogram")
                                outputHistEq2 = gr.Plot(label="Equalized Histogram")
                                outputHistClahe2 = gr.Plot(label="CLAHE Histogram")
                                outputHistMatch2 = gr.Plot(label="Matched Histogram")

            observation = gr.Textbox(
                label="Observation",
                lines=7
            )

            runBtn.click(
                fn=task7,
                inputs=[
                    inputImage1,
                    inputImage2,
                    referenceImage
                ],
                outputs=[

                    outputGray1,
                    outputEqualized1,
                    outputClahe1,
                    outputMatch1,

                    outputHistGray1,
                    outputHistEq1,
                    outputHistClahe1,
                    outputHistMatch1,

                    outputGray2,
                    outputEqualized2,
                    outputClahe2,
                    outputMatch2,

                    outputHistGray2,
                    outputHistEq2,
                    outputHistClahe2,
                    outputHistMatch2,

                    observation
                ]
            )

demo.launch()