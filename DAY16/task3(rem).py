import os
from ultralytics import YOLO

MODEL_PATH="runs/detect/runs/task3_training/weights/best.pt"
TEST_FOLDER="images_task2/images/test"
IMAGE_SIZE=640
THRESHOLDS=[0.25,
    0.50,
    0.75
]

CLASS_NAMES=["bottle",
    "book",
    "dog"
]

if not os.path.exists(MODEL_PATH):
    print("ERROR: Trained model was not found!")
    print()
    print("Expected model:")
    print(MODEL_PATH)
    exit()

print()
print("Trained model found:")
print(MODEL_PATH)

if not os.path.exists(TEST_FOLDER):
    print()
    print("ERROR: Test folder was not found!")
    print()
    print("Expected test folder:")
    print(TEST_FOLDER)
    exit()

test_images=[file
    for file in os.listdir(TEST_FOLDER)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

if len(test_images)==0:
    print()
    print("ERROR: No test images found!")
    exit()

print()
print("UNSEEN TEST IMAGES")
print()
print("Test images found:", len(test_images))
for image in test_images:
    print("-", image)

print()
print("LOADING TRAINED YOLO MODEL")
print()
model=YOLO(MODEL_PATH)
print()
print("Model class names:")
print(model.names)
print("Model loaded successfully!")

os.makedirs("inference_results", exist_ok=True)
comparison_results=[]
for threshold in THRESHOLDS:
    print()
    print(f"CONFIDENCE THRESHOLD: {threshold}")
    print()

    folder_name=(f"confidence_{int(threshold * 100)}")
    predictions=model.predict(source=TEST_FOLDER,
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
        if boxes is None or len(boxes)==0:
            print(f"Image: " f"{os.path.basename(result.path)}")
            print("No objects detected.")
            print()
            continue
        print(f"Image: " f"{os.path.basename(result.path)}")
        for i in range(len(boxes)):
            class_id=int(boxes.cls[i].item())
            confidence=float(boxes.conf[i].item())
            if class_id<len(CLASS_NAMES):
                class_name=CLASS_NAMES[class_id]
            else:
                class_name="unknown"
            total_detections+=1
            print(f"  Object {i + 1}: " f"{class_name}")
            print(f"  Confidence: " f"{confidence:.2f}")
        print()

    comparison_results.append((threshold, total_detections))
    print("Total detections:", total_detections)
    print("Results saved in:")
    print(os.path.join("inference_results", folder_name))

print()
print("CONFIDENCE THRESHOLD COMPARISON")
print()
for threshold, detections in comparison_results:
    print(f"Threshold {threshold:.2f} "
        f"-> {detections} detections")

print()
print("Trained model:")
print(MODEL_PATH)

print()
print("Unseen test images:")
print(TEST_FOLDER)

print()
print("Prediction results:")
print("inference_results/")

print()
print("Confidence thresholds tested:")
print("0.25")
print("0.50")
print("0.75")

print()
print("Classes:")
print("0 = bottle")
print("1 = book")
print("2 = dog")

print()
print("EFFECT OF CONFIDENCE THRESHOLD")

print()
print("0.25:")
print("Lower threshold allows more detections.")
print("It may detect more objects but can also "
    "produce more false positives.")

print()
print("0.50:")
print("Provides a more balanced threshold.")
print("It removes some low-confidence detections "
    "while keeping reasonably confident objects.")

print()
print("0.75:")
print("Higher threshold keeps only highly "
    "confident detections.")
print("It can reduce false positives but may "
    "increase false negatives by removing "
    "real objects with lower confidence.")
#The trained YOLO model was tested on unseen test images using different confidence thresholds (0.25, 0.50, and 0.75). The model generated candidate predictions for the test images, but their confidence scores were relatively low. Therefore, after applying the selected 
#confidence thresholds, no final bounding boxes were displayed.
#This experiment demonstrates how confidence 
# thresholds affect YOLO predictions. A lower threshold generally allows more predictions to appear, while a higher threshold removes low-confidence predictions and can increase false negatives. In this case, the model's predictions on the selected unseen images were below the tested thresholds.