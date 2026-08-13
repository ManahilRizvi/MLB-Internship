import numpy as np
import matplotlib.pyplot as plt
def homogeneous(point):
    x=point[0]
    y=point[1]
    homPoint=np.array([x, y, 1])
    return homPoint

def multiplicationMatrix(matrix, point):
    result=[0, 0, 0]
    for i in range(3):
        total=0
        for j in range(3):
            total=total+(matrix[i][j]*point[j])
        result[i]=total
    return result

def convert2D(point):
    x=point[0]
    y=point[1]
    w=point[2]

    x=x/w
    y=y/w
    return[round(float(x), 2), round(float(y), 2)]

points=[[3, 3],
        [6, 2],
        [4, 3],
        [6, 6],
        [9, 8]]

print("Points: ")
print(points)

homMatrix=[[1, 0, 0],
           [0, 1, 0],
           [0.1, 0.1, 1]]
print("Homography Matrix: ")
print(homMatrix)

transPoints=[]
for point in points:
    homPoint=homogeneous(point)
    transformHom=multiplicationMatrix(homMatrix, homPoint)
    tp=convert2D(transformHom)
    transPoints.append(tp)

print("Transformed Points: ")
for point in transPoints:
    print(point)

ogX=[]
ogY=[]
transX=[]
transY=[]
for point in points:
    ogX.append(point[0])
    ogY.append(point[1])

for point in transPoints:
    transX.append(point[0])
    transY.append(point[1])

ogX.append(ogX[0])
ogY.append(ogY[0])
transX.append(transX[0])
transY.append(transY[0])

plt.figure(figsize=(8, 6))
plt.plot(ogX, ogY, marker="o", label="Original Points")
plt.plot(transX, transY, marker="o", label="Transformed Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Perspective Transformation")
plt.legend()
plt.grid()
plt.show()