import cv2
import matplotlib.pyplot as plt
#reading image in grayscale mode
img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image")
else:
    rows, columns=img.shape#getting height and width
    
    #no of coordinates to mark
    noOfCod=int(input("\nCoordinates to mark: "))
    plt.imshow(img, cmap="gray")
    #taking coordinates from user
    for i in range(noOfCod):
        print("Coordinate", i+1)
        row=int(input("\nEnter row: "))
        col=int(input("\nEnter column: "))
        #checking if values are valid
        if row>=0 and row<rows and col>=0 and col<columns:
            print("Value of Pixel: ", img[row][col])
            #marking coordinate
            plt.scatter(col, row, color="red", s=40)
            #displaying coordinated text
            plt.text(col+5, row+5, "(" +str(col)+","+str(row)+")", color="yellow", fontsize=8)
        else:
            print("Invalid")
    
    plt.title("Coordinates of Image")
    plt.axis("off")
    plt.show()

#Image coordinate system uses(x, y)
#x represents horizontal position(column)
#y represents vertical position(row)
#Matrix indexing uses(row, column) 
#row represents vertical position
#column represents horizontal position
#both systems refer to same pixel but order od values are diiferent
#img coordinate=(40, 20)
#matrix index=img[20][40]