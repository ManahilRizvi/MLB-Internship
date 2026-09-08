import os
import shutil
import random
from ultralytics import YOLO

IMAGE_SOURCE="images_task2\images"
LABEL_SOURCE="images_task2\labels"
DATASET_FOLDER="images_task2"

TRAIN_RATIO=0.70
VAL_RATIO=0.15
TEST_RATIO=0.15
EPOCHS=30
IMAGE_SIZE=640
CLASS_NAMES=["bottle", "book", "dog"]

if TRAIN_RATIO+VAL_RATIO+TEST_RATIO!=1.0:
    raise ValueError("Train, validation and test ratios must add up to 1.0")

folders=[os.path.join(DATASET_FOLDER, "images", "train"),
    os.path.join(DATASET_FOLDER, "images", "val"),
    os.path.join(DATASET_FOLDER, "images", "test"),
    os.path.join(DATASET_FOLDER, "labels", "train"),
    os.path.join(DATASET_FOLDER, "labels", "val"),
    os.path.join(DATASET_FOLDER, "labels", "test")]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

image_files=[file for file in os.listdir(IMAGE_SOURCE)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))]
if len(image_files)==0:
    print("No images found!")
    print("Make sure your images are inside:")
    print(IMAGE_SOURCE)
    exit()

print()
print("Total images found:", len(image_files))

valid_images=[]
for image_file in image_files:
    image_name=os.path.splitext(image_file)[0]
    label_file=image_name+".txt"
    label_path=os.path.join(LABEL_SOURCE, label_file)
    if os.path.exists(label_path):
        valid_images.append(image_file)
    else:
        print("Warning: Label not found for", image_file)

print("Images with labels:", len(valid_images))
if len(valid_images)==0:
    print("No matching image-label pairs found.")
    exit()

random.seed(42)
random.shuffle(valid_images)

total=len(valid_images)
train_count=int(total * TRAIN_RATIO)
val_count=int(total * VAL_RATIO)
test_count=total-train_count-val_count

train_images=valid_images[:train_count]
val_images=valid_images[train_count:train_count+val_count]
test_images=valid_images[train_count+val_count:]

print()
print("Dataset split:")
print("Training images:", len(train_images))
print("Validation images:", len(val_images))
print("Testing images:", len(test_images))

def copy_dataset(image_list, split):
    image_destination=os.path.join(DATASET_FOLDER, "images", split)
    label_destination=os.path.join(DATASET_FOLDER, "labels", split)
    for image_file in image_list:
        image_source_path=os.path.join(IMAGE_SOURCE, image_file)
        image_destination_path=os.path.join(image_destination, image_file)
        shutil.copy2(image_source_path, image_destination_path)

        image_name=os.path.splitext(image_file)[0]
        label_file=image_name+".txt"
        label_source_path=os.path.join(LABEL_SOURCE, label_file)
        label_destination_path=os.path.join(label_destination, label_file)
        shutil.copy2(label_source_path, label_destination_path)

print()
print("Copying training data...")
copy_dataset(train_images, "train")

print("Copying validation data...")
copy_dataset(val_images, "val")

print("Copying test data...")
copy_dataset(test_images, "test")
print("Dataset prepared successfully...")

yaml_path=os.path.join(DATASET_FOLDER, "data.yaml")
absolute_dataset_path=os.path.abspath(DATASET_FOLDER)
yaml_content=f"""path: {absolute_dataset_path}
train: images/train
val: images/val
test: images/test
names:
  0: bottle
  1: book
  2: dog
nc: 3
"""

with open(yaml_path, "w") as file:
    file.write(yaml_content.strip())

print()
print("data.yaml created:")
print(yaml_path)
print()
print("STARTING YOLO TRAINING")
print()

model=YOLO("yolo26n.pt")
results=model.train(data=yaml_path,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=8,
    project="runs",
    name="task3_training",
    patience=10,
    plots=True
)
print()
print("TRAINING COMPLETED")

best_model_path=os.path.join("runs",
    "detect",
    "runs",
    "task3_training",
    "weights",
    "best.pt")

if not os.path.exists(best_model_path):
    print()
    print("Best model was not found!")
    print("Check the runs folder.")
    exit()

print()
print("Best model:")
print(best_model_path)

trained_model=YOLO(best_model_path)
print()
print("VALIDATING MODEL")
print()

metrics=trained_model.val(data=yaml_path)
print()
print("Validation completed...")

thresholds=[0.25, 0.50, 0.75]
for threshold in thresholds:
    folder_name=(f"confidence_{int(threshold * 100)}")
    os.makedirs(
        os.path.join("inference_results", folder_name), exist_ok=True)
    
test_folder=os.path.join(DATASET_FOLDER,
    "images",
    "test"
)
print()
print("RUNNING INFERENCE")
print()

for threshold in thresholds:
    print()
    print(f"Confidence threshold: {threshold}")
    folder_name=(f"confidence_{int(threshold * 100)}")
    save_folder=os.path.join("inference_results", folder_name)
    predictions=trained_model.predict(source=test_folder,
        conf=threshold,
        imgsz=IMAGE_SIZE,
        save=True,
        project="inference_results",
        name=folder_name,
        exist_ok=True,
        verbose=False
    )
    total_detections=0
    for result in predictions:
        boxes=result.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            class_id=int(boxes.cls[i].item())
            confidence=float(boxes.conf[i].item())
            class_name=CLASS_NAMES[class_id]
            total_detections+=1
            print(
                f"Image: "
                f"{os.path.basename(result.path)}"
            )
            print(f"Class: {class_name}")
            print(f"Confidence: " f"{confidence:.2f}")
            print()
    print(f"Total detections at "
        f"confidence {threshold}: "
        f"{total_detections}"
    )

print()
print("Training results:")
print("runs/task3_training/")

print()
print("Test predictions:")
print("inference_results/")

print()
print("Confidence levels tested:")
print("0.25")
print("0.50")
print("0.75")

print()
print("Classes:")
print("0 = bottle")
print("1 = book")
print("2 = dog")