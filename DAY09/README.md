# 🖼️ Image Noise, Noise Removal & Blur Analysis

An interactive Image Processing project developed using **Python, NumPy, OpenCV, Matplotlib, and Gradio**.

This project demonstrates the manual implementation of image noise generation, noise removal, blur generation, blur detection, and blur analysis without relying on built-in image processing functions. A Gradio web application is also included to provide an interactive interface for testing all tasks.

---

## 📌 Features

### ✅ Task 1 – Gaussian Noise Generation
- Generate Gaussian Noise for Grayscale Images
- Generate Gaussian Noise for RGB Images
- Manual RGB to Grayscale Conversion
- Adjustable Gaussian Noise (Mean = 0, Standard Deviation = 20)

---

### ✅ Task 2 – Noise Removal
Applied noise reduction techniques on noisy images.

Implemented Filters:
- Mean Filter (Manual)
- Gaussian Filter (Manual)

Comparison between:
- Original Image
- Noisy Image
- Mean Filter Output
- Gaussian Filter Output

---

### ✅ Task 3 – Blur Generation

Implemented different blur techniques manually.

### Gaussian Blur
- 3×3 Kernel
- 5×5 Kernel
- 7×7 Kernel

### Motion Blur
- 3×3 Kernel
- 5×5 Kernel
- 7×7 Kernel

### Median Blur
- 3×3 Kernel
- 5×5 Kernel
- 7×7 Kernel

---

### ✅ Task 4 – Blur Detection

Implemented manual **Variance of Laplacian** for blur detection.

Steps:
- Manual RGB to Grayscale Conversion
- Manual Laplacian Convolution
- Mean Calculation
- Variance Calculation
- Blur Classification

Image Categories:
- Sharp Image
- Blurry Image

---

### ✅ Task 5 – Blur Metrics & Analysis

Evaluated multiple images using the Variance of Laplacian.

Generated:
- Blur Score
- Sharp / Blurry Classification
- Observation Table
- Brief Analysis of Results

---

# 🛠 Technologies Used

- Python
- NumPy
- OpenCV
- Matplotlib
- Gradio

---

# 📂 Project Structure

```
DAY09/
│
├── gradio_demo.py
├── task1.py
├── task2.py
├── task3.py
├── task4.py
├── task5.py
│
├── cat.jpeg
├── dog.jpg
├── clear.jpeg
├── images.jpeg
├── mouse.jpeg
├── road.jpeg
├── river.jpeg
├── lowCon.jpg
├── orange-flower.jpg
├── jpeg_43-2.jpg
│
└── README.md
```
---

# 📖 Learning Outcomes

Through this project, I learned:

- Manual implementation of Gaussian Noise
- Image smoothing using Mean and Gaussian Filters
- Working with Gaussian, Motion, and Median Blur
- Blur Detection using Variance of Laplacian
- Manual convolution using kernels
- Image quality analysis
- Building interactive Gradio applications

---

# 👩‍💻 Author

**Manahil Rizvi**
