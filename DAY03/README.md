# Module 2: Color Spaces - Image Processing

This repository contains implementations of different **Color Space concepts** using Python, OpenCV, NumPy, and Matplotlib.

The main focus of these tasks is to understand how images are represented in different color spaces and how color transformations help in image processing applications.
---

# 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib

---

## ✅ Task 1: RGB and BGR Understanding

### Description:
In this task, an image is read using OpenCV and displayed in both **BGR and RGB formats**.

The conversion from BGR to RGB is implemented manually by swapping the Red and Blue channels.

### Concepts Learned:
- OpenCV image representation
- Difference between RGB and BGR
- Manual channel swapping
- Image visualization

---

## ✅ Task 2: Split and Merge Color Channels

### Description:
An RGB image is separated into individual color channels:

- Red Channel
- Green Channel
- Blue Channel

After displaying each channel separately, the channels are merged again to reconstruct the original image.

### Concepts Learned:
- Image channels
- Channel separation
- Channel merging
- RGB image structure

---

## ✅ Task 3: HSV Color Segmentation

### Description:
A colored object is detected using HSV color space.

The RGB image is manually converted into HSV, then thresholding is applied to create a mask and extract the required object.

### Concepts Learned:
- HSV color representation
- Hue, Saturation, and Value
- Color detection
- Image masking
- Object segmentation

---

## ✅ Task 4: LAB Color Analysis

### Description:
The image is converted into LAB color space manually.

The three LAB channels are extracted and visualized separately:

- L Channel → Lightness
- A Channel → Green to Red information
- B Channel → Blue to Yellow information

### Concepts Learned:
- RGB to XYZ conversion
- XYZ to LAB conversion
- LAB color space
- Color and brightness separation

---

## ✅ Task 5: Color Space Conversion

### Description:
The same image is converted into multiple color spaces:

- RGB
- HSV
- HSL
- LAB
- Grayscale

All converted outputs are displayed together for comparison.

### Concepts Learned:
- Different image representations
- Purpose of different color spaces
- Manual color conversion
- Image visualization
