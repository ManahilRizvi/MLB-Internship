# YOLO Object Detection — Custom Dataset

## Project Overview

This project implements an object detection system using **YOLO** on a custom annotated dataset.

The model was trained to detect three object classes:

* Bottle
* Book
* Dog

The project covers the complete object detection workflow, including bounding boxes, IoU, dataset preparation, YOLO annotation, model training, evaluation, inference, Non-Maximum Suppression (NMS), and a Gradio web demo with ngrok.

---

# 1. Dataset

A custom dataset was created containing images of three object classes:

| Class ID | Class Name |
| -------- | ---------- |
| 0        | Bottle     |
| 1        | Book       |
| 2        | Dog        |

The dataset was divided into:

* Training set
* Validation set
* Test set

The dataset follows the YOLO directory structure:

```text
images_task2/
│
├── data.yaml
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

The `data.yaml` file contains the dataset paths and class names.

---

# 2. YOLO Annotations

Each image has a corresponding `.txt` annotation file.

YOLO annotation format is:

```text
class_id x_center y_center width height
```

The coordinates are normalized between `0` and `1`.

For example:

```text
0 0.52 0.48 0.30 0.55
```

This represents a bounding box for class `0` (Bottle).

The annotation format allows YOLO to understand both the object class and its location in the image.

---

# 3. Bounding Boxes and IoU

Bounding boxes were defined around objects in the images.

Different bounding-box formats were studied and converted into the YOLO normalized format.

### IoU — Intersection over Union

IoU measures the overlap between a predicted bounding box and a ground-truth bounding box.

```text
IoU = Area of Intersection / Area of Union
```

IoU values range from `0` to `1`.

* **IoU = 0:** The two boxes do not overlap.
* **IoU = 1:** The two boxes are exactly the same.

IoU was implemented from scratch using Python without relying on a ready-made IoU function.

---

# 4. Annotation Visualization

The annotated dataset was visualized to verify that the bounding boxes were correctly placed around the objects.

This step was useful for checking annotation quality before training the YOLO model.

The dataset contains annotations for:

* Bottles
* Books
* Dogs

---

# 5. YOLO Model Training

A pretrained YOLO model was used as the starting point for custom object detection.

The model was trained on the custom dataset for **30 epochs**.

Training was performed using the prepared YOLO dataset and `data.yaml`.

The training process generated:

* Training and validation losses
* Precision
* Recall
* mAP
* Precision-Recall curves
* F1 curves
* Confusion matrices
* Best model checkpoint

The best trained model was saved as:

```text
runs/detect/runs/task3_training/weights/best.pt
```

The `best.pt` checkpoint was used for evaluation and inference.

---

# 6. Training and Validation Results

The best validation results obtained from the trained model were:

| Metric    | Result |
| --------- | ------ |
| Precision | 92.18% |
| Recall    | 66.12% |
| mAP@50    | 78.94% |
| mAP@50:95 | 33.77% |

### Observation

The model achieved high precision, meaning that most of its confident detections were correct.

The recall was lower than precision, indicating that the model missed some objects.

The mAP@50 result shows that the model was able to detect the objects reasonably well at the IoU 0.50 criterion.

The lower mAP@50:95 compared with mAP@50 indicates that accurate bounding-box localization becomes more difficult when stricter IoU thresholds are used.

---

# 7. Inference on Test Images

The trained `best.pt` model was used to perform inference on unseen test images.

Different confidence thresholds were tested:

```text
0.25
0.50
0.75
```

The purpose was to observe how changing the confidence threshold affects the number of displayed detections.

### Observation

The selected unseen test images produced relatively low-confidence predictions.

At the tested confidence thresholds, many predictions were filtered out and no final bounding boxes were displayed on the selected images.

A diagnostic test using a lower confidence threshold showed that the model was producing candidate predictions, but many of their confidence scores were below `0.25`.

This experiment demonstrated that:

* A lower confidence threshold allows more predictions.
* A higher confidence threshold removes more low-confidence predictions.
* Increasing the confidence threshold can reduce false positives but may increase false negatives.

---

# 8. Non-Maximum Suppression (NMS)

Non-Maximum Suppression is used to remove duplicate bounding boxes for the same object.

YOLO may initially generate multiple overlapping predictions for one object.

NMS compares:

* Confidence scores
* Bounding-box overlap

The prediction with the higher confidence is kept, while highly overlapping duplicate predictions are suppressed.

---

# 9. Confidence Threshold vs NMS IoU Threshold

These two thresholds have different purposes.

### Confidence Threshold

The confidence threshold filters predictions according to their confidence score.

For example:

```text
Confidence = 0.05
Confidence threshold = 0.25
```

The prediction will be removed because its confidence is below `0.25`.

### NMS IoU Threshold

The NMS IoU threshold controls how much two bounding boxes can overlap before one of them is suppressed.

* Lower IoU threshold → more aggressive suppression
* Higher IoU threshold → more overlapping boxes are allowed

Therefore:

> **Confidence threshold filters predictions based on confidence, while NMS IoU threshold controls overlapping duplicate predictions.**

---

# 10. NMS Experiment

NMS was tested using different IoU thresholds:

```text
0.30
0.50
0.70
```

The confidence threshold was kept fixed at:

```text
0.25
```

### Observation

No final detections were produced on the selected unseen test images at any of the tested NMS IoU thresholds.

This occurred because the candidate predictions on these images had relatively low confidence and were filtered by the confidence threshold before meaningful differences in NMS could be observed.

In general, changing the NMS IoU threshold affects how overlapping bounding boxes are suppressed.

---

# 11. Why Accuracy Alone Is Not Sufficient

Accuracy alone is not sufficient for object detection because object detection involves two important tasks:

1. Correctly identifying the object class.
2. Correctly locating the object using a bounding box.

A prediction can have the correct class but an inaccurate bounding box.

Therefore, object detection is evaluated using metrics such as:

* Precision
* Recall
* mAP@50
* mAP@50:95

These metrics provide more useful information about false positives, missed detections, and bounding-box localization quality.

---

# 12. Gradio Demo

A Gradio web interface was created using the trained `best.pt` model.

The demo allows the user to upload an image and run YOLO object detection.

The output displays:

* Bounding boxes
* Class names
* Confidence scores

The Gradio interface uses a confidence threshold of:

```text
0.01
```

### Why 0.01 was used in the demo

The trained model produced relatively low-confidence predictions on some unseen images.

Using a confidence threshold of `0.25` filtered out many of these predictions.

Therefore, a lower threshold of `0.01` was used in the demonstration so that low-confidence predictions could also be displayed.

A lower threshold may also display predictions that are less certain.

---

# 13. ngrok Public Demo

The Gradio application was exposed to the internet using **ngrok**.

The Gradio application was first started locally:

```bash
python task5.py
```

The local application runs on:

```text
http://127.0.0.1:7860
```

Then ngrok was started using:

```bash
ngrok http 7860
```

ngrok generates a public HTTPS URL that can be opened in a browser.

The public URL was used to test the Gradio YOLO detection demo.

---

# 14. Project Files

The main project structure is:

```text
DAY16/
│
├── task1.py
├── task2.py
├── task3.py
├── task3_inference.py
├── task4.py
├── task5.py
├── README.md
│
├── images_task2/
│   ├── data.yaml
│   │
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── gradio_examples/
│   ├── bottle9.jpg
│   ├── book8.jpg
│   ├── dog7.jpg
│   ├── book1.jpg
│   ├── dog1.jpg
│   ├── dog10.jpg
│   ├── book11.jpg
│   ├── bottle5.jpg
│   └── bottle1.jpg
│
└── runs/
    └── detect/
        └── runs/
            └── task3_training/
                └── weights/
                    ├── best.pt
                    └── last.pt
```

---

# 15. Approach Summary

The project followed these steps:

```text
Collect Images
      ↓
Annotate Objects
      ↓
Convert Annotations to YOLO Format
      ↓
Create Train / Validation / Test Splits
      ↓
Create data.yaml
      ↓
Train Pretrained YOLO Model
      ↓
Save Best Model
      ↓
Evaluate Model
      ↓
Run Inference on Test Images
      ↓
Experiment with Confidence Thresholds
      ↓
Experiment with NMS IoU Thresholds
      ↓
Create Gradio Demo
      ↓
Expose Demo Using ngrok
```

---

# 16. Overall Observations

The project demonstrated the complete workflow of custom object detection using YOLO.

The model achieved a **Precision of 92.18%** and **mAP@50 of 78.94%**, showing that it learned to detect the three object classes.

However, the recall of **66.12%** shows that some objects were missed, particularly on the selected unseen test images.

The experiments with confidence thresholds demonstrated that increasing the confidence threshold removes more low-confidence predictions.

The NMS experiment demonstrated the purpose of controlling overlapping predictions, although meaningful differences between NMS thresholds were not visible on the selected test images because their predictions were already below the confidence threshold.

Finally, a Gradio interface was created to provide an interactive way to test the trained YOLO model, and ngrok was used to make the application accessible through a public URL.

---
* [x] README.md
