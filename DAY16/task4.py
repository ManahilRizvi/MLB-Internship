import os
from ultralytics import YOLO

MODEL_PATH="runs/detect/runs/task3_training/weights/best.pt"
DATA_YAML="images_task2/data.yaml"
TEST_FOLDER="images_task2/images/test"
IMAGE_SIZE=640
NMS_THRESHOLDS=[0.30, 0.50, 0.70]
CONFIDENCE_THRESHOLD=0.25

if not os.path.exists(MODEL_PATH):
    print("Best model was not found!")
    print(MODEL_PATH)
    exit()

print()
print("LOADING TRAINED YOLO MODEL")
print()
model=YOLO(MODEL_PATH)
print("Model loaded successfully!")
print(MODEL_PATH)

print()
print("MODEL EVALUATION")
print()

metrics=model.val(data=DATA_YAML)
print()
print("Evaluation completed...")

precision=float(metrics.box.mp)
recall=float(metrics.box.mr)
map50=float(metrics.box.map50)
map50_95=float(metrics.box.map)

print()
print("EVALUATION METRICS")
print()
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"mAP@50         : {map50:.4f}")
print(f"mAP@50:95      : {map50_95:.4f}")

os.makedirs("task4_nms_results", exist_ok=True)
print()
print("NMS IoU THRESHOLD EXPERIMENT")
print()
nms_results=[]
for nms_iou in NMS_THRESHOLDS:
    folder_name=f"iou_{int(nms_iou * 100)}"
    save_folder=os.path.join("task4_nms_results", folder_name)
    os.makedirs(save_folder, exist_ok=True)
    print(f"NMS IoU threshold: {nms_iou}")
    predictions=model.predict(source=TEST_FOLDER,
        conf=CONFIDENCE_THRESHOLD,
        iou=nms_iou,
        imgsz=IMAGE_SIZE,
        save=True,
        project="task4_nms_results",
        name=folder_name,
        exist_ok=True,
        verbose=False
    )

    total_detections=0
    for result in predictions:
        boxes=result.boxes
        if boxes is None:
            continue
        number_of_boxes=len(boxes)
        total_detections+=number_of_boxes
        if number_of_boxes==0:
            print(os.path.basename(result.path), "-> No final detections")
        else:
            print(os.path.basename(result.path),
                "->",
                number_of_boxes,
                "final detections")
            for i in range(number_of_boxes):
                class_id=int(boxes.cls[i].item())
                confidence=float(boxes.conf[i].item())
                class_name=model.names[class_id]
                print(f"   {class_name} "
                    f"(confidence = {confidence:.2f})"
                )

    nms_results.append((nms_iou, total_detections))
    print("Total final detections:", total_detections)
    print()

print()
print("NMS THRESHOLD COMPARISON")
print()

for threshold, detections in nms_results:
    print(f"NMS IoU {threshold:.2f} "
        f"-> {detections} final detections")

print()
print("Precision      :", f"{precision:.4f}")
print("Recall         :", f"{recall:.4f}")
print("mAP@50         :", f"{map50:.4f}")
print("mAP@50:95      :", f"{map50_95:.4f}")

print()
print("NMS IoU thresholds tested:")
print("0.30")
print("0.50")
print("0.70")

print()
print("Results saved in:")
print("task4_nms_results/")
### NMS — Non-Maximum Suppression
#Non-Maximum Suppression (NMS) is used in object detection to 
# remove duplicate bounding boxes for the same object. YOLO may 
# initially generate multiple overlapping predictions for one 
# object. NMS compares their confidence scores and the overlap 
# between their bounding boxes. The prediction with the higher 
# confidence is kept, while highly overlapping duplicate predictions are suppressed.

### Confidence Threshold vs IoU Threshold
#A confidence threshold determines whether a prediction is confident 
# enough to be considered a final detection. Increasing the confidence 
# threshold removes more low-confidence predictions.

#An IoU threshold in NMS controls how much two bounding boxes can 
# overlap before one of them is suppressed. A lower NMS IoU threshold 
# is more aggressive in removing overlapping boxes, while a higher IoU 
# threshold allows more overlapping boxes to remain.
#Therefore, confidence threshold controls the quality/confidence of predictions, 
# while NMS IoU threshold controls duplicate overlapping predictions.

### Evaluation Results
#The trained YOLO model achieved the following validation results:
#Precision: 92.2%
#Recall: 66.1%
#mAP@50: 78.9%
#mAP@50:95: 33.8%

### Why Accuracy Alone Is Not Sufficient
#Accuracy alone is not sufficient for object detection because object 
#detection has two important tasks: identifying the correct object class 
# and locating the object with an accurate bounding box.
#A prediction can have the correct class but an incorrect 
# bounding box. Accuracy does not properly describe this 
# localization quality. Precision, Recall, and mAP provide 
# more useful information about detection performance, including 
# false positives, missed objects, and bounding-box overlap with ground-truth annotations.
#Therefore, object detection models are evaluated using metrics 
#such as Precision, Recall, mAP@50, and mAP@50:95 rather than relying only on accuracy.

#### Observation
#The trained YOLO model was evaluated using Precision, 
# Recall, mAP@50, and mAP@50:95. The model achieved a Precision 
# of 92.18%, Recall of 66.12%, mAP@50 of 78.94%, and mAP@50:95 of 33.77%.
#NMS was also tested using IoU thresholds of 0.30, 0.50, and 0.70. 
# On the selected unseen test images, no final detections were produced
# at any of these thresholds. This was because the model's candidate predictions 
# had relatively low confidence scores and were filtered by the confidence threshold 
# of 0.25 before meaningful NMS differences could be observed.
#In general, a lower NMS IoU threshold is more aggressive 
# at removing overlapping bounding boxes, while a higher IoU 
# threshold allows more overlapping boxes to remain. The confidence 
# threshold and NMS IoU threshold perform different functions: confidence 
# threshold filters predictions based on their confidence scores, whereas NMS 
# IoU threshold controls the suppression of overlapping duplicate bounding boxes.
#Accuracy alone is not sufficient for object detection because 
# detection requires both correct object classification and accurate 
# localization. Precision, Recall, and mAP provide a better evaluation 
# of object detection performance because they consider correct detections,
#  missed objects, false detections, and bounding-box overlap.