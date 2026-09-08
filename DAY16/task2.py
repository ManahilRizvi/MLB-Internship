import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button

IMAGE_FOLDER="images_task2\images"
LABEL_FOLDER="images_task2\labels"
classes={0: "bottle", 1: "book", 2: "dog"}
os.makedirs(LABEL_FOLDER, exist_ok=True)

image_files=[file for file in os.listdir(IMAGE_FOLDER)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]
image_files.sort()
if len(image_files)==0:
    print("No images found!")
    print("Put your images inside the 'images_task2' folder.")
    exit()
print("Total images:", len(image_files))

current_image_index=0
current_image=None
boxes=[]
current_box=None
fig=None
ax=None
selector=None

def load_image():
    global current_image
    image_path=os.path.join(IMAGE_FOLDER, image_files[current_image_index])
    current_image=plt.imread(image_path)
    current_image=np.array(current_image)
    return current_image

def on_select(click, release):
    global current_box
    if click.xdata is None or click.ydata is None:
        return
    if release.xdata is None or release.ydata is None:
        return

    x1=min(click.xdata, release.xdata)
    y1=min(click.ydata, release.ydata)
    x2=max(click.xdata, release.xdata)
    y2=max(click.ydata, release.ydata)
    current_box=(x1, y1, x2, y2)
    print()
    print("Bounding box created!")
    print("Now click class button:")
    print("BOTTLE / BOOK / DOG")

def add_box(class_id):
    global current_box
    if current_box is None:
        print()
        print("Please draw bounding box first!")
        return

    x1, y1, x2, y2=current_box
    boxes.append({"class_id": class_id, "box": (x1, y1, x2, y2)})
    print()
    print("Object added!")
    print("Class:", classes[class_id])
    print("Total objects in image:", len(boxes))
    ax.plot([x1, x2, x2, x1, x1],
        [y1, y1, y2, y2, y1],
        linewidth=2)
    ax.text(x1,
        y1,
        classes[class_id],
        fontsize=10,
        backgroundcolor="white")
    current_box=None
    fig.canvas.draw()

def bottle_clicked(event):
    add_box(0)

def book_clicked(event):
    add_box(1)

def dog_clicked(event):
    add_box(2)

def convert_to_yolo():
    height, width=current_image.shape[:2]
    yolo_lines=[]
    for item in boxes:
        class_id=item["class_id"]
        x1, y1, x2, y2=item["box"]
        box_width=x2-x1
        box_height=y2-y1
        x_center=x1+(box_width/2)
        y_center=y1+(box_height/2)
        x_center=x_center/width
        y_center=y_center/height
        box_width=box_width/width
        box_height=box_height/height

        line=(f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}")
        yolo_lines.append(line)
    return yolo_lines

def save_labels():
    if len(boxes)==0:
        print()
        print("No objects were annotated in this image.")
        return
    yolo_lines=convert_to_yolo()
    image_name=image_files[current_image_index]
    label_name=os.path.splitext(image_name)[0]+".txt"
    label_path=os.path.join(LABEL_FOLDER, label_name)
    with open(label_path, "w") as file:
        for line in yolo_lines:
            file.write(line+"\n")

    print()
    print("Labels saved!")
    print("File:", label_path)
    print("Objects:", len(boxes))

    print()
    print("YOLO annotations:")
    for line in yolo_lines:
        print(line)

def finish_image(event):
    save_labels()
    print()
    print("Current image finished.")
    print("You can click NEXT IMAGE.")

def next_image(event):
    global current_image_index
    global boxes
    global current_box
    save_labels()
    current_image_index+=1
    if current_image_index>=len(image_files):
        print()
        print("ALL IMAGES COMPLETED!")
        plt.close()
        return

    boxes=[]
    current_box=None
    show_image()

def show_image():
    global selector
    ax.clear()
    image=load_image()
    ax.imshow(image)
    ax.set_title(f"Image {current_image_index + 1} / {len(image_files)}\n"
        "Draw a box around an object, then click its class button")
    ax.axis("off")
    selector=RectangleSelector(ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True)
    fig.canvas.draw_idle()

fig, ax=plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.18)
bottle_ax=plt.axes([0.10, 0.04, 0.14, 0.07])
bottle_button=Button(bottle_ax, "BOTTLE")
bottle_button.on_clicked(bottle_clicked)

book_ax=plt.axes([0.28, 0.04, 0.14, 0.07])
book_button=Button(book_ax, "BOOK")
book_button.on_clicked(book_clicked)

dog_ax=plt.axes([0.46, 0.04, 0.14, 0.07])
dog_button=Button(dog_ax, "DOG")
dog_button.on_clicked(dog_clicked)

finish_ax=plt.axes([0.64, 0.04, 0.14, 0.07])
finish_button=Button(finish_ax, "FINISH")
finish_button.on_clicked(finish_image)

next_ax=plt.axes([0.82, 0.04, 0.14, 0.07])
next_button=Button(next_ax, "NEXT")
next_button.on_clicked(next_image)
show_image()
plt.show()