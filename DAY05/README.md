# Image Thresholding & Histogram Processing (Manual Implementation)

This repository contains the manual implementation of Image Thresholding and Histogram Processing techniques using **Python**, **OpenCV**, **NumPy**, and **Matplotlib**.

The objective of this task is to understand how thresholding and histogram-based image enhancement techniques work internally by implementing them manually wherever possible, instead of relying on OpenCV's built-in functions.

---

## Tasks Completed

### ✅ Task 1: Manual Image Thresholding

Implemented the following thresholding techniques **without using `cv2.threshold()`**:

- Binary Thresholding
- Binary Inverse Thresholding
- Truncate Thresholding
- To Zero Thresholding
- Inverse To Zero Thresholding

---

### ✅ Task 2: Histogram Analysis

- Converted image to grayscale manually.
- Calculated histogram manually.
- Plotted histogram using Matplotlib.
- Compared histograms of dark, bright, and normal images.
- Analyzed image brightness and contrast using histogram distribution.

---

### ✅ Task 3: Manual Histogram Equalization

Implemented Histogram Equalization **without using `cv2.equalizeHist()`**.

Steps included:

- Manual Histogram Calculation
- Cumulative Distribution Function (CDF)
- CDF Normalization
- Equalized Image Generation
- Comparison of Original and Equalized Images

---

### ✅ Task 4: CLAHE Comparison

Applied CLAHE using OpenCV and compared its results with Manual Histogram Equalization using different types of images:

- Dark Image
- Bright Image
- Low Contrast Image

---


## 🛠 Technologies Used

- Python 3
- OpenCV (cv2)
- NumPy
- Matplotlib



BS Computer Science Student | Learning Image Processing & Computer Vision
