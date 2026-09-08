import os
import gradio as gr
from ultralytics import YOLO

MODEL_PATH="runs/detect/runs/task3_training/weights/best.pt"
IMAGE_FOLDER="gradio_examples"
CONFIDENCE_THRESHOLD=0.01
model=YOLO(MODEL_PATH)
print("YOLO model loaded successfully!")
print("Classes:", model.names)

def detect_objects(image):
    if image is None:
        return None
    results=model.predict(source=image,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=640,
        verbose=False
    )
    result=results[0]
    output_image=result.plot()
    return output_image

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # YOLO Object Detection Demo

        Upload one of the prepared test images from the
        **`gradio_examples`** folder.

        The YOLO model detects:

        - Bottle
        - Book
        - Dog

        The detection result displays:

        - Bounding boxes
        - Class names
        - Confidence scores

        ---

        ### Why is the confidence threshold set to 0.01?

        The trained YOLO model produces relatively low confidence
        scores on some unseen images. With a confidence threshold
        of **0.25**, many predictions are filtered out and no
        bounding boxes may be displayed.

        Therefore, a lower threshold of **0.01** is used for this
        demonstration so that low-confidence predictions can also
        be displayed.

        **Note:** A lower confidence threshold may display
        predictions that are less certain.
        """
    )

    gr.Markdown(
        """
        ## Upload Test Image

        Please select one of the 9 prepared images from the
        **`gradio_examples`** folder.
        """
    )

    input_image=gr.Image(type="numpy",
        sources=["upload"],
        label="Upload Image"
    )

    detect_button=gr.Button("Run YOLO Detection")
    output_image=gr.Image(type="numpy", label="Detection Result")
    detect_button.click(fn=detect_objects, inputs=input_image, outputs=output_image)

demo.launch(server_name="127.0.0.1", server_port=7860)