import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import gradio as gr


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "wasteCnn.pth")
EXAMPLES_PATH = os.path.join(BASE_DIR, "examples")

IMG_SIZE = 128

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SAVED GRAPH FILES
# ============================================================

LOSS_GRAPH = os.path.join(BASE_DIR, "finalmodelLoss_Graph.png")
ACC_GRAPH = os.path.join(BASE_DIR, "finalmodelAcc_Graph.png")
CONFUSION_MATRIX = os.path.join(BASE_DIR, "confusionMatrix.png")
EXAMPLES_PRED = os.path.join(BASE_DIR, "examplesPred.png")
LEARNING_RATE = os.path.join(BASE_DIR, "learningRateGraph.png")
OPTIMIZER_GRAPH = os.path.join(BASE_DIR, "sgd&adamValAcc_Graph.png")


# ============================================================
# SAME CNN USED IN final_project.py
# ============================================================

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3
        )

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            2,
            2
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            2,
            2
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            28800,
            128
        )

        self.fcRelu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(
            128,
            6
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fcRelu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

evalTrans = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )

])


# ============================================================
# LOAD MODEL
# ============================================================

model = SimpleCNN().to(device)

model_loaded = False

if os.path.exists(MODEL_PATH):

    try:

        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=device,
                weights_only=True
            )
        )

        model.eval()

        model_loaded = True

        print("Model loaded successfully.")
        print("Model:", MODEL_PATH)

    except Exception as e:

        print("Model loading error:")
        print(e)

else:

    print("Model file not found:")
    print(MODEL_PATH)


# ============================================================
# EXAMPLE IMAGES
# ============================================================

def get_example_images():

    if not os.path.exists(EXAMPLES_PATH):
        return []

    files = []

    for filename in sorted(
        os.listdir(EXAMPLES_PATH)
    ):

        if filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            )
        ):

            files.append(
                os.path.join(
                    EXAMPLES_PATH,
                    filename
                )
            )

    return files


example_images = get_example_images()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(img):

    if img is None:

        return (
            "No image selected.",
            "",
            "Please upload or select an example image."
        )

    if not model_loaded:

        return (
            "Model not loaded.",
            "",
            "Make sure wasteCnn.pth is in the same folder as this app."
        )

    try:

        # Convert input into PIL image
        if not isinstance(img, Image.Image):

            img = Image.fromarray(
                np.array(img)
            )

        img = img.convert("RGB")

        # Apply same preprocessing used during evaluation
        input_img = evalTrans(
            img
        ).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():

            output = model(input_img)

            probabilities = F.softmax(
                output,
                dim=1
            )[0]

        confidence, predicted = torch.max(
            probabilities,
            0
        )

        predicted_class = CLASSES[
            predicted.item()
        ]

        confidence_value = (
            confidence.item() * 100
        )

        # ----------------------------------------------------
        # Prediction result
        # ----------------------------------------------------

        result_text = (
            f"Predicted Class: {predicted_class.upper()}\n\n"
            f"Confidence: {confidence_value:.2f}%"
        )

        # ----------------------------------------------------
        # Probability of every class
        # ----------------------------------------------------

        probability_lines = []

        for i, class_name in enumerate(CLASSES):

            probability = (
                probabilities[i].item() * 100
            )

            probability_lines.append(
                f"{class_name.title():<12} : {probability:.2f}%"
            )

        probability_text = "\n".join(
            probability_lines
        )

        status_text = (
            "Prediction completed successfully."
        )

        return (
            result_text,
            probability_text,
            status_text
        )

    except Exception as e:

        return (
            "Prediction error.",
            "",
            f"Error: {str(e)}"
        )


# ============================================================
# GRAPH / IMAGE AVAILABILITY
# ============================================================

def image_exists(path):
    return os.path.isfile(path)


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="Smart Waste Classification CNN"
) as app:

    # ========================================================
    # MAIN TITLE
    # ========================================================

    gr.Markdown(
        """
# ♻️ Smart Waste Classification Using CNN

### Classify waste images into 6 categories using a PyTorch CNN

**Categories:** Cardboard • Glass • Metal • Paper • Plastic • Trash
"""
    )


    # ========================================================
    # TAB 1 - PREDICTION
    # ========================================================

    with gr.Tab("🔍 Prediction"):

        gr.Markdown(
            """
## Upload or Select a Waste Image

Upload your own image, or select one of the images
from the **examples** folder.
"""
        )

        with gr.Row():

            # ------------------------------------------------
            # LEFT SIDE
            # ------------------------------------------------

            with gr.Column(scale=1):

                image_input = gr.Image(
                    type="pil",
                    label="Waste Image",
                    height=350
                )

                predict_button = gr.Button(
                    "🔎 Predict",
                    variant="primary"
                )

            # ------------------------------------------------
            # RIGHT SIDE
            # ------------------------------------------------

            with gr.Column(scale=1):

                prediction_output = gr.Textbox(
                    label="Prediction Result",
                    lines=4
                )

                probability_output = gr.Textbox(
                    label="Class Probabilities",
                    lines=8
                )

                status_output = gr.Textbox(
                    label="Status",
                    lines=2
                )


        # ----------------------------------------------------
        # EXAMPLES
        # ----------------------------------------------------

        gr.Markdown(
            "## 📸 Example Images"
        )

        if example_images:

            gr.Examples(
                examples=[
                    [image_path]
                    for image_path in example_images
                ],
                inputs=image_input,
                label="Click an image to select it",
                examples_per_page=10
            )

        else:

            gr.Markdown(
                """
⚠️ **No example images found.**

Please create an `examples` folder in the same directory
as this Gradio file and put your sample images inside it.
"""
            )


        # ----------------------------------------------------
        # PREDICT BUTTON
        # ----------------------------------------------------

        predict_button.click(
            fn=predict_image,
            inputs=image_input,
            outputs=[
                prediction_output,
                probability_output,
                status_output
            ]
        )


        # ----------------------------------------------------
        # ALSO PREDICT WHEN IMAGE CHANGES
        # ----------------------------------------------------

        image_input.change(
            fn=predict_image,
            inputs=image_input,
            outputs=[
                prediction_output,
                probability_output,
                status_output
            ]
        )


    # ========================================================
    # TAB 2 - GRAPHS & IMAGES
    # ========================================================

    with gr.Tab("📊 Graphs & Images"):

        gr.Markdown(
            """
# 📊 Model Performance

The following images are the graphs and results generated
during the CNN training and evaluation process.
"""
        )


        # ----------------------------------------------------
        # FINAL MODEL LOSS
        # ----------------------------------------------------

        gr.Markdown(
            "## 📉 Final Model — Training & Validation Loss"
        )

        if image_exists(LOSS_GRAPH):

            gr.Image(
                value=LOSS_GRAPH,
                label="Final Model Loss Graph",
                show_label=True
            )

        else:

            gr.Markdown(
                "⚠️ `finalmodelLoss_Graph.png` not found."
            )


        # ----------------------------------------------------
        # FINAL MODEL ACCURACY
        # ----------------------------------------------------

        gr.Markdown(
            "## 📈 Final Model — Training & Validation Accuracy"
        )

        if image_exists(ACC_GRAPH):

            gr.Image(
                value=ACC_GRAPH,
                label="Final Model Accuracy Graph",
                show_label=True
            )

        else:

            gr.Markdown(
                "⚠️ `finalmodelAcc_Graph.png` not found."
            )


        # ----------------------------------------------------
        # SGD VS ADAM
        # ----------------------------------------------------

        gr.Markdown(
            "## ⚙️ Optimizer Comparison — SGD vs Adam"
        )

        if image_exists(OPTIMIZER_GRAPH):

            gr.Image(
                value=OPTIMIZER_GRAPH,
                label="SGD vs Adam Validation Accuracy",
                show_label=True
            )

        else:

            gr.Markdown(
                "⚠️ `sgd&adamValAcc_Graph.png` not found."
            )


        # ----------------------------------------------------
        # LEARNING RATE
        # ----------------------------------------------------

        gr.Markdown(
            "## 📊 Learning Rate Experiment"
        )

        if image_exists(LEARNING_RATE):

            gr.Image(
                value=LEARNING_RATE,
                label="Learning Rate Comparison",
                show_label=True
            )

        else:

            gr.Markdown(
                "⚠️ `learningRateGraph.png` not found."
            )


        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        gr.Markdown(
            "## 🔲 Confusion Matrix"
        )

        if image_exists(CONFUSION_MATRIX):

            gr.Image(
                value=CONFUSION_MATRIX,
                label="Confusion Matrix",
                show_label=True
            )

        else:

            gr.Markdown(
                "⚠️ `confusionMatrix.png` not found."
            )


        # ----------------------------------------------------
        # EXAMPLE PREDICTIONS
        # ----------------------------------------------------

        gr.Markdown(
            "## 🖼️ Predictions on Example Images"
        )

        if image_exists(EXAMPLES_PRED):

            gr.Image(
                value=EXAMPLES_PRED,
                label="Example Image Predictions",
                show_label=True
            )

        else:

            gr.Markdown(
                "⚠️ `examplesPred.png` not found."
            )


    # ========================================================
    # TAB 3 - REPORT
    # ========================================================

    with gr.Tab("📄 Report"):

        gr.Markdown(
            """
# ♻️ Smart Waste Classification Using Convolution Neural Network

*A simple CNN trained in PyTorch for automated waste image sorting*

---

## 1. Introduction

This project implements Convolution Neural Network (CNN) to classify waste images into 6 categories: **cardboard, glass, metal, paper, plastic and trash**. The goal is to automate waste sorting using deep learning which can support recycling systems and reduce manual sorting effort.

---

## 2. Dataset

The dataset used contains **2527 images** distributed across 6 classes:

- **Cardboard:** 403 images
- **Glass:** 501 images
- **Metal:** 410 images
- **Paper:** 594 images
- **Plastic:** 482 images
- **Trash:** 137 images

All images were resized to **128×128 pixels** and normalized before being fed into the model.

The dataset was split into:

- **Training set:** 1768 images (70%)
- **Validation set:** 379 images (15%)
- **Testing set:** 380 images (15%)

Data augmentation (random horizontal flip and rotation) was applied to the training set to improve generalization and reduce overfitting.

---

## 3. Model Architecture

A simple CNN was designed consisting of:

- **Convolution Layer 1:** 16 filters, 3×3 kernel + ReLU + MaxPooling
- **Convolution Layer 2:** 32 filters, 3×3 kernel + ReLU + MaxPooling
- **Fully Connected Layer:** 128 neurons + ReLU + Dropout (0.5)
- **Output Layer:** 6 neurons, one per class

The model was trained using **CrossEntropyLoss** as the loss function.

---

## 4. Experiments

### 4.1 Optimizer Comparison (SGD vs Adam)

Both optimizers were trained for 10 epochs at learning rate **0.001** to compare convergence speed and accuracy.

- **SGD:** Final Train Accuracy = **27.49%** | Final Val Accuracy = **23.75%**
- **Adam:** Final Train Accuracy = **68.61%** | Final Val Accuracy = **63.06%**

Adam clearly outperformed SGD, converging much faster and achieving significantly higher accuracy within the same number of epochs. This is expected since Adam uses adaptive learning rates while plain SGD without momentum converges slowly on this type of dataset.

---

### 4.2 Learning Rate Experiment

Two learning rates were tested with the Adam optimizer:

- **LR = 0.001:** Final Train Accuracy = **70.76%** | Final Val Accuracy = **66.75%**
- **LR = 0.0005:** Final Train Accuracy = **63.97%** | Final Val Accuracy = **63.06%**

---

## 5. Final Model Training

The final model was trained using **Adam optimizer with a learning rate of 0.0005 for 10 epochs**.

### Final Epoch Results

- **Train Loss:** 0.8527
- **Train Accuracy:** 69.57%
- **Validation Loss:** 0.9181
- **Validation Accuracy:** 67.55%

Training and validation loss decreased consistently across epochs and accuracy improved steadily without significant divergence between training and validation curves, indicating that the model did not overfit significantly.

---

## 6. Final Evaluation (Accuracy Report)

The trained model was evaluated on an unseen test set of **380 images**.

| Metric | Score |
|---|---:|
| Test Accuracy | **67.37%** |
| Precision | **66.90%** |
| Recall | **67.37%** |
| F1 Score | **65.56%** |

*Precision, Recall and F1 score were computed using weighted average across all 6 classes.*

---

## 6.1 Confusion Matrix

| Actual / Predicted | Cardboard | Glass | Metal | Paper | Plastic | Trash |
|---|---:|---:|---:|---:|---:|---:|
| **Cardboard** | 47 | 3 | 2 | 5 | 2 | 0 |
| **Glass** | 3 | 43 | 6 | 8 | 15 | 3 |
| **Metal** | 1 | 5 | 37 | 15 | 3 | 0 |
| **Paper** | 1 | 2 | 2 | 86 | 1 | 0 |
| **Plastic** | 4 | 5 | 2 | 15 | 40 | 0 |
| **Trash** | 5 | 5 | 4 | 5 | 2 | 3 |

### Observations

- **Paper** was classified with the highest accuracy (86 correct out of 92), likely due to having the largest number of training samples.
- **Trash** had the lowest accuracy, which is expected since it had the fewest training samples (137 images) and visually more diverse content.
- **Glass and Plastic** were sometimes confused with each other, which is reasonable given their similar transparent or reflective appearance.

---

## 7. Predictions on New Images

The trained model was tested on **7 new previously unseen sample images** stored in the `examples` folder.

It correctly classified **5 out of 7 images (71.4%)**, which is consistent with the overall test accuracy of 67%.

The two misclassifications (glass predicted as paper and plastic predicted as paper) both occurred with low confidence scores (under 41%), indicating that the model itself was uncertain in these cases.

These errors are consistent with confusion patterns observed in the confusion matrix where visually similar items were occasionally misclassified.

---

## 8. Conclusion

CNN model achieved a test accuracy of approximately **67%**, which is a reasonable result given the simplicity of the architecture (only 2 convolution layers), the moderate dataset size and training being performed on CPU.

Adam optimizer significantly outperformed SGD and learning rate of **0.0005** provided stable convergence for the final model.

Misclassifications mainly occurred between visually similar categories (glass/plastic) and the under-represented trash class.

Future improvements could include using a deeper CNN architecture, transfer learning with a pretrained model and collecting more training data for under-represented classes such as trash.
"""
        )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SMART WASTE CLASSIFICATION - GRADIO APP")
    print("=" * 60)

    print("Device:", device)
    print("Model loaded:", model_loaded)
    print("Example images:", len(example_images))

    print()
    print("Graph files:")

    print(
        "Loss:",
        image_exists(LOSS_GRAPH)
    )

    print(
        "Accuracy:",
        image_exists(ACC_GRAPH)
    )

    print(
        "Optimizer:",
        image_exists(OPTIMIZER_GRAPH)
    )

    print(
        "Learning Rate:",
        image_exists(LEARNING_RATE)
    )

    print(
        "Confusion Matrix:",
        image_exists(CONFUSION_MATRIX)
    )

    print(
        "Example Predictions:",
        image_exists(EXAMPLES_PRED)
    )

    print()
    print("Open the local Gradio URL shown below.")
    print()

    app.launch()