import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
img=cv2.imread("task2(diagram).png")
rows, columns, channel=img.shape
rgb=np.zeros_like(img)
for i in range(rows):
    for j in range(columns):
        blue=img[i][j][0]
        green=img[i][j][1]
        red=img[i][j][2]
        rgb[i][j][0]=red
        rgb[i][j][1]=green
        rgb[i][j][2]=blue
print("-----COMPONENTS OF NEURAL NETWORK-----")
print("Input Layer: Contains input values x1, x2 and x3")
print("Hidden Layer: Contains neurons h1, h2 and h3. It process input data and learns patterns")
print("Output Layer: Produces final output or prediction")
print("Weights: Weights w11, w12, w13, w21, ... determine how important each input is to neuron")
print("Bias: Bias is additional value added to weighted sum. It helps neuron adjust its output")

print("\n FLOW OF DATA.........")
print("Data flows through neural networks from input layer to output layer.")
print("First input values are passed to neurons in hidden layer. Each input")
print("is multiplied by its corresponding weight. These weight values are")
print("added together along with bias. Formula is: z=x1w1+x2w2+x3w3+b")
print("Calculated value is then passed through activation function which")
print("determines output of neuron. Outputs of hidden layer neurons are")
print("then passed to output neuron. Output neuron performs similar calculation")
print("using its weight and bias and produces final prediction.")
figure, ax=plt.subplots(figsize=(6, 8))
boxes=[("Input Data", 0.25, 0.82),
       ("Weights+Bias", 0.25, 0.65),
       ("Hidden Layer", 0.25, 0.48),
       ("Activation Function", 0.25, 0.31),
       ("Output Layer", 0.25, 0.14),
       ("Final Prediction", 0.25, -0.03)]

for text, x, y in boxes:
    box=Rectangle((x, y), 0.5, 0.09, fill=False, linewidth=2)
    ax.add_patch(box)
    ax.text(x+0.25, y+0.045, text, ha="center", va="center", fontsize=12)
for i in range(len(boxes)-1):
    x=0.5
    y1=boxes[i][2]
    y2=boxes[i+1][2]+0.09 
    ax.annotate("", xy=(x, y2), xytext=(x, y1), arrowprops=dict(arrowstyle="->", linewidth=1.5))
ax.set_xlim(0, 1)
ax.set_ylim(-0.1, 1)
plt.axis("off")
plt.title("Flow of Data through Neural Network", fontsize=14)
plt.show()

plt.imshow(rgb)
plt.axis("off")
plt.show()