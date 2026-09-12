# License Plate Detection Model

A YOLOv8-based object detection model trained to detect license plates in real-world images, including challenging conditions like low light, blur, and angled views. This project focuses on **detection only** (localizing the plate), not reading the plate number.

## Project Overview

1. **Dataset**: [License Plate Recognition dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e) (v12, raw/unaugmented, 10,125 images) downloaded via Roboflow in YOLOv8 format.
2. **Exploration**: Images were analyzed and categorized into 4 conditions based on brightness and blur — `day_sharp`, `night_sharp`, `day_blurry`, `night_blurry`.
3. **Balancing**: The dataset was rebalanced using oversampling (minority classes) and undersampling (majority class) to reduce the imbalance between clear daytime images and rarer night/blurry images.
4. **Augmentation**: Applied using Albumentations — rotation, brightness/contrast adjustment, Gaussian blur, Gaussian noise, random cropping, and partial occlusion (CoarseDropout) — to help the model generalize to real-world edge cases.
5. **Training**: A YOLOv8n model was trained on the balanced + augmented dataset (13,875 images) for 15 epochs on a Tesla T4 GPU (Google Colab).
6. **Testing**: The trained model was evaluated separately on normal images and edge-case images (blurry/low-light) from the test set.
7. **Deployment**: A Gradio web app was built for interactive testing and deployed publicly using ngrok.

## Results

| Metric | Score |
|---|---|
| Precision | 98.05% |
| Recall | 94.18% |
| mAP50 | 96.14% |
| mAP50-95 | 67.87% |

The model was tested on 723 normal images and 297 edge-case images (blurry/night/angled), successfully detecting license plates across both sets, including in low-light and low-quality conditions.

## Files

- `license_plate_detection.ipynb` — Full notebook: dataset download, exploration, balancing, augmentation, training, testing, and Gradio app code.
- `weights/best.pt` — Trained YOLOv8 model weights.
- `screenshots/` — Sample outputs from the deployed Gradio app showing detection on normal and edge-case images.

## Dataset & Full Project Data

Full dataset (raw, balanced, and augmented versions) is hosted on Google Drive due to size constraints:
**[Insert Google Drive Link Here]**

## How to Run

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model.predict("path/to/image.jpg", conf=0.25)
results[0].show()
```

## Gradio App

The notebook includes a Gradio interface for uploading an image and viewing detected license plates in real time, deployed publicly via ngrok during development.
[(https://drive.google.com/drive/folders/YOUR_FOLDER_ID)](https://drive.google.com/drive/folders/1b_PMqZu-kajs5KiRshirKwWlwKVXYBYT?usp=sharing)
