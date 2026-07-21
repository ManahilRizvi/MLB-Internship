import math
#asking user how many coordinate pairs they want
pairsCod=int(input("Enter no of pairs of coordinate: "))

#repeating process for each pair
for i in range(pairsCod):
    print("\nCoordinate Pair", i+1)

    #first coordinate
    x1=int(input("Enter x1: "))
    y1=int(input("Enter y1: "))
    #second coordinate
    x2=int(input("Enter x2: "))
    y2=int(input("Enter y2: "))
    #difference between x coordinates
    dx=x2-x1
    #difference between y coordinates
    dy=y2-y1
    #calculating euclidean distance
    distance=(dx*dx+dy*dy)**0.5
    #verifying result
    mathDist=math.sqrt((dx*dx)+(dy*dy))

    print("Euclidean Distance: ", distance)
    print("Python's math function: ", mathDist)