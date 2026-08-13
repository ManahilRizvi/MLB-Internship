import numpy as np
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

point=[5, 7]
print("Point: ", point)
homPoint=homogeneous(point)
print("Homogeneous Point: ", homPoint)
transMatrix=[[1, 0, 4],
            [0, 1, 2],
            [0, 0, 1]]
transPoint=multiplicationMatrix(transMatrix, homPoint)
print("After Translation: ", [int(x) for x in transPoint])

scaleMatrix=[[2, 0, 0],
             [0, 3, 0],
             [0, 0, 1]]
scalePoint=multiplicationMatrix(scaleMatrix, homPoint)
print("After Scaling: ", [int(x) for x in scalePoint])

theta=np.radians(90)
rotationMatrix=[[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]]
rotatePoint=multiplicationMatrix(rotationMatrix, homPoint)
print("After Rotation: ", [int(x) for x in rotatePoint])
