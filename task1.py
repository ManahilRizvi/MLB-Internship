import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def xyxyToxywh(box):
    xMin, yMin, xMax, yMax=box
    xCenter=(xMin+xMax)/2
    yCenter=(yMin+yMax)/2
    width=xMax-xMin
    height=yMax-yMin
    return [xCenter, yCenter, width, height]

def xywhToxyxy(box):
    xCenter, yCenter, width, height=box
    xMin=xCenter-width/2
    yMin=yCenter-height/2
    xMax=xCenter+width/2
    yMax=yCenter+height/2
    return [xMin, yMin, xMax, yMax]

def xywhToyolo(box, imgWidth, imgHeight, classId):
    xCenter, yCenter, width, height=box
    xCenter=xCenter/imgWidth
    yCenter=yCenter/imgHeight
    width=width/imgWidth
    height=height/imgHeight
    return [classId, xCenter, yCenter, width, height]

def calculateIou(box1, box2):
    x1Min, y1Min, x1Max, y1Max=box1
    x2Min, y2Min, x2Max, y2Max=box2
    intersectionXmin=max(x1Min, x2Min)
    intersectionYmin=max(y1Min, y2Min)
    intersectionXmax=min(x1Max, x2Max)
    intersectionYmax=min(y1Max, y2Max)
    intersectionWidth=max(0, intersectionXmax-intersectionXmin)
    intersectionHeight=max(0, intersectionYmax-intersectionYmin)
    intersectionArea=(intersectionWidth*intersectionHeight)
    areaBox1=((x1Max-x1Min)*(y1Max-y1Min))
    areaBox2=((x2Max-x2Min)*(y2Max-y2Min))
    unionArea=areaBox1+areaBox2-intersectionArea
    if unionArea==0:
        return 0
    iou=intersectionArea/unionArea
    return iou

drawing=False
startPoint=None
endPoint=None
currImg=None

def boxDrawing(event, x, y, flags, param):
    global drawing
    global startPoint
    global endPoint
    global currImg
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        startPoint=(x, y)
        endPoint=(x, y)
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing:
            endPoint=(x, y)
            tempImg=currImg.copy()
            cv2.rectangle(tempImg, startPoint, endPoint, (0, 0, 255), 2)
            cv2.imshow("Draw Bounding Box", tempImg)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        endPoint=(x, y)
        cv2.rectangle(currImg, startPoint, endPoint, (0, 0, 255), 2)
        cv2.imshow("Draw Bounding Box", currImg)

def selectBox(image):
    global currImg
    global startPoint
    global endPoint
    currImg=image.copy()
    startPoint=None
    endPoint=None
    windowName="Draw Bounding Box"
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, boxDrawing)
    print("\nDRAW BOX AROUND OBJECT...")
    print("press ENTER after selecting box...")
    print("press ESC to skip image")
    while True:
        cv2.imshow(windowName, currImg)
        key=cv2.waitKey(1)&0xFF
        if key==13:
            if startPoint is not None and endPoint is not None:
                x1=min(startPoint[0], endPoint[0])
                y1=min(startPoint[1], endPoint[1])
                x2=max(startPoint[0], endPoint[0])
                y2=max(startPoint[1], endPoint[1])
                box=[x1, y1, x2, y2]
                cv2.destroyWindow(windowName)
                return box
        elif key==27:
            cv2.destroyWindow(windowName)
            return None

imgFolder="images_task1"
imgFiles=[]
for name in os.listdir(imgFolder):
    if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        imgFiles.append(name)
imgFiles.sort()
print("\nImages: ", len(imgFiles))
results=[]
for index, name in enumerate(imgFiles):
    imgPath=os.path.join(imgFolder, name)
    img=cv2.imread(imgPath)
    if img is None:
        print("No image...", name)
        continue
    print("\nImage: ", index+1, "/", len(imgFiles))
    print("Name: ", name)
    imgHeight, imgWidth=img.shape[:2]
    print("Img Width: ", imgWidth)
    print("Img Height: ", imgHeight)
    xyxyBox=selectBox(img)
    if xyxyBox is None:
        print("Skipped Image...")
        continue
    xywhBox=xyxyToxywh(xyxyBox)
    classId=0
    yoloBox=xywhToyolo(xywhBox, imgWidth, imgHeight, classId)
    xyxyConvert=xywhToxyxy(xywhBox)
    print("\nXYXY: ")
    print(xyxyBox)
    print("\nXYWH: ")
    print(xywhBox)
    print("\nYOLO Normalized: ")
    print(yoloBox)
    print("\nXYWH to XYXY: ")
    print(xyxyConvert)
    output={"image": name, 
            "width": imgWidth,
            "height": imgHeight,
            "xyxy": xyxyBox,
            "xywh": xywhBox,
            "yolo": yoloBox}
    results.append(output)

print("\n IOU TESTS")
box1=[100, 100, 300, 300]
box2=[100, 100, 300, 300]
iou1=calculateIou(box1, box2)
print("Test on Same Boxes...")
print("Box 1: ", box1)
print("Box 2: ", box2)
print("IOU: ", iou1)

box1=[50, 50, 150, 150]
box2=[200, 200, 300, 300]
iou2=calculateIou(box1, box2)
print("Test on No Overlapping Boxes...")
print("Box 1: ", box1)
print("Box 2: ", box2)
print("IOU: ", iou2)

box1=[100, 100, 300, 300]
box2=[200, 200, 400, 400]
iou3=calculateIou(box1, box2)
print("Test on Partial Overlapping Boxes...")
print("Box 1: ", box1)
print("Box 2: ", box2)
print("IOU: ", iou3)

box1=[100, 100, 400, 400]
box2=[200, 200, 300, 300]
iou4=calculateIou(box1, box2)
print("Test on One Box Inside Another...")
print("Box 1: ", box1)
print("Box 2: ", box2)
print("IOU: ", iou4)

print("Final Outputs...")
for result in results:
    print("\nImage: ", result["image"])
    print("XYXY: ", result["xyxy"])
    print("XYWH: ", result["xywh"])
    print("YOLO: ", result["yolo"])

#iou=0...
#when two bounding boxes donot overlap their intersection area is 0
#iou=1...
#when two bounding boxes are exactly same their intersection area and union area are equal