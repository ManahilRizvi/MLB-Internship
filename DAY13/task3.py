import numpy as np
def matrixCreate(srcPoints, destPoints):
    A=[]
    for i in range(len(srcPoints)):
        x=srcPoints[i][0]
        y=srcPoints[i][1]
        u=destPoints[i][0]
        v=destPoints[i][1]

        row1=[-x, -y, -1,
              0, 0, 0,
              u*x, u*y, u]

        row2=[0, 0, 0,
              -x, -y, -1,
              v*x, v*y, v]

        A.append(row1)
        A.append(row2)
    return np.array(A, dtype=float)

def dlt(srcPoints, destPoints):
    if len(srcPoints)<4:
        print("4 points are required...")
        return None
    A=matrixCreate(srcPoints, destPoints)
    U, S, Vt=np.linalg.svd(A)
    h=Vt[-1]
    H=h.reshape(3, 3)
    H=H/H[2][2]
    return H

def homography(H, point):
    x=point[0]
    y=point[1]
    homPoint=[x, y, 1]
    result=[0, 0, 0]
    for i in range(3):
        total=0
        for j in range(3):
            total=total+H[i][j]*homPoint[j]
        result[i]=total
    if result[2]==0:
        return None
    newX=result[0]/result[2]
    newY=result[1]/result[2]
    return[round(float(newX), 2), round(float(newY), 2)]

def errorCalculate(H, srcPoint, destPoint):
    transPoint=homography(H, srcPoint)
    if transPoint is None:
        return float("inf")
    diffX=transPoint[0]-destPoint[0]
    diffY=transPoint[1]-destPoint[1]
    error=np.sqrt(diffX*diffX+diffY*diffY)
    return error

def ransac(srcPoints, destPoints, iterations=100, threshold=2):
    numPoints=len(srcPoints)
    bestH=None
    bestInliers=[]
    for iteration in range(iterations):
        indices=np.random.choice(numPoints, 4, replace=False)
        sampleSrc=[]
        sampleDest=[]
        for index in indices:
            sampleSrc.append(srcPoints[index])
            sampleDest.append(destPoints[index])
        H=dlt(sampleSrc, sampleDest)
        if H is None:
            continue
        currentInliers=[]
        for i in range(numPoints):
            error=errorCalculate(H, srcPoints[i], destPoints[i])
            if error<threshold:
                currentInliers.append(i)
        if len(currentInliers)>len(bestInliers):
            bestInliers=currentInliers
            bestH=H
    return bestH, bestInliers

srcPoints=[[0, 0],
           [10, 0],
           [10, 10],
           [0, 10],
           [5, 5],
           [2, 8]]

destPoints=[[2, 3],
            [12, 3],
            [12, 13],
            [2, 13],
            [7, 8],
            [20, 20]]

srcPoint1=srcPoints[0]
srcPoint2=srcPoints[1]
srcPoint3=srcPoints[2]
srcPoint4=srcPoints[3]

destPoint1=destPoints[0]
destPoint2=destPoints[1]
destPoint3=destPoints[2]
destPoint4=destPoints[3]

allSrc=[srcPoint1, srcPoint2, srcPoint3, srcPoint4]
allDest=[destPoint1, destPoint2, destPoint3, destPoint4]
H=dlt(allSrc, allDest)
print("Using DLT: ")
print(H)
print("Transformed Points: ")
for point in allSrc:
    transPoint=homography(H, point)
    print(point, ": ", transPoint)

bestH, inliers=ransac(srcPoints, destPoints, iterations=100, threshold=2)
print("Using RANSAC: ")
print(bestH)
print("Inlier Indices: ")
print(inliers)
print("No of Inliers: ")
print(len(inliers))