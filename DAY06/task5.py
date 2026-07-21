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
    #converting negative into positive values
    if dx<0:
        dx=-dx
    if dy<0:
        dy=-dy
    #calculating manhattan distance
    manhdistance=dx+dy

    print("Euclidean Distance: ", distance)
    print("manhattan Distance: ", manhdistance)

    #comparing distance
    if distance<manhdistance:
        print("Euclidean Distance is smaller")
    elif distance>manhdistance:
        print("Manhattan Distance is smaller")
    else:
        print("Both Distances Equal")
#euclidean distance is suitable when shortest straight line
#distance is required

#manhattan distance is suitable when movement is limited
#to horizontal and vertical direction such as city roads,
#grid maps and robot movement