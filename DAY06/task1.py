import cv2
import matplotlib.pyplot as plt
#reading image in grayscale mode
img=cv2.imread("cat.jpeg", 0)
if img is None:
    print("No image")
else:
    rows, columns=img.shape#getting height and width
    print("Matrix of image: \n")#printing matrix of image
    print(img)

    print("\nHeight: ", rows)
    print("Width: ", columns)

    #getting image coordinates from user
    row=int(input("\nEnter row no: "))
    col=int(input("\nEnter column no: "))
    #checking whether user input is inside image or not
    if row>=0 and row<rows and col>=0 and col<columns:
        #pixel value at that row and column
        print("Pixel value at (", row, ",", col, ")=", img[row][col])
    else:
        print("Invalid")
    
    #asking which pixel want to modify
    row=int(input("\nEnter row for modification: "))
    col=int(input("\nEnter column for modification: "))
    #user enter new pixel value
    pixVal=int(input("Enter new values of pixel(0 to 255): "))
    
    #checking if values are valid
    if row>=0 and row<rows and col>=0 and col<columns and pixVal>=0 and pixVal<=255:
        #replacing old value with new value
        img[row][col]=pixVal
        print("Pixels Update Done...")
    else:
        print("Invalid")
    

    plt.imshow(img, cmap="gray")
    plt.title("Image Updated")
    plt.axis("off")
    plt.show()