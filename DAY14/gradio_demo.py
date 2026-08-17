import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
def task1():
    output=""
    output+="------ARTIFICIAL INTELLIGENCE(AI)--------\n\n"
    output+="AI allows computers and machines to perform tasks which"
    output+="normally requires human intelligence. They can make decisions,"
    output+="understand information, solve problems and interact with people."
    output+="REAL WORLD EXAMPLES OF AI: \n"
    output+="1. Smart Traffic Systems\n"
    output+="2. Game AI\n"
    output+="3. Virtual Assistants\n\n\n"

    output+="\n------MACHINE LEARNING (ML)--------\n\n"
    output+="ML is a branch of AI in which computers learn patterns from"
    output+="data and use these patterns to make predictions or decisions."
    output+="We provide data so that system can learn from it instead of"
    output+="giving computer instructions for every situation.\n\n"
    output+="REAL WORLD EXAMPLES OF ML: \n"
    output+="1. Recommendation Systems\n"
    output+="2. House Price Prediction Systems\n"
    output+="3. Spam Email Detection\n\n\n"

    output+="\n------DEEP LEARNING (DL)--------\n\n"
    output+="DL is a branch of ML that uses artficial neural networks with"
    output+="multiple layers. They can learn complex patterns from large amounts"
    output+="of data with less need for manually designing features.\n\n"
    output+="REAL WORLD EXAMPLES OF DL: \n"
    output+="1. Medical Image Analysis\n"
    output+="2. Object Detection\n"
    output+="3. Face Recognition\n\n\n"

    output+="\n--------------------------------------------------------------------"
    output+="WHY IS DL USEFUL FOR COMPUTER VISION?\n\n"
    output+="DL is very useful for computer vision because image contain"
    output+="large amount of complex information. Traditional methods often"
    output+="require humans to manually design features such as edges, shapes"
    output+="and textures. DL models especially CNN (Convolutional Neural Networks)"
    output+="can automatically learn useful features from images. In early years"
    output+="network can learn simple features such as edges and textures. In deeper layers"
    output+="it can learn more complex features such as shapes, objects"
    output+="and patterns. That's why it is useful for many computer vision applications"
    output+="Such as: \n"
    output+="1. Medical Image Analysis\n"
    output+="2. Object Detection\n"
    output+="3. Face Recognition\n"
    output+="4. Image Classification\n"
    output+="5. Self Driving Vehicle Vision Systems"
    return output

def task2():
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
    output=""
    output+="-----COMPONENTS OF NEURAL NETWORK-----\n\n"
    output+="Input Layer: Contains input values x1, x2 and x3\n"
    output+="Hidden Layer: Contains neurons h1, h2 and h3. It process input data and learns patterns\n"
    output+="Output Layer: Produces final output or prediction\n"
    output+="Weights: Weights w11, w12, w13, w21, ... determine how important each input is to neuron\n"
    output+="Bias: Bias is additional value added to weighted sum. It helps neuron adjust its output\n"

    output+="\n FLOW OF DATA.........\n\n"
    output+="Data flows through neural networks from input layer to output layer.\n"
    output+="First input values are passed to neurons in hidden layer. Each input\n"
    output+="is multiplied by its corresponding weight. These weight values are\n"
    output+="added together along with bias. Formula is: z=x1w1+x2w2+x3w3+b\n"
    output+="Calculated value is then passed through activation function which\n"
    output+="determines output of neuron. Outputs of hidden layer neurons are\n"
    output+="then passed to output neuron. Output neuron performs similar calculation\n"
    output+="using its weight and bias and produces final prediction."
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
    ax.set_title("Flow of Data through Neural Network", fontsize=14)
    plt.tight_layout()
    return rgb, figure, output

def task3():
    input=np.array([2, 3, 1])
    weight=np.array([0.5, 0.2, 0.8])
    bias=0.5
    weightSum=0
    for i in range(len(input)):
        weightSum=weightSum+(input[i]*weight[i])
    weightSum=weightSum+bias
    print("Weighted Sum: ", weightSum)
    if weightSum>0:
        output=weightSum
    else:
        output=0
    result=""
    result+="Inputs: "+str(input)+"\n"
    result+="Weights: "+str(weight)+"\n"
    result+="Bias: "+str(bias)+"\n\n"
    result+="Weighted Sum: "+str(weightSum)+"\n"
    result+="Activation Function: ReLU\n"
    result+="Output: "+str(output)
    return result

def task4():
    data=np.array([[1, 20],
                [2, 42],
                [3, 46],
                [4, 85],
                [5, 17],
                [6, 82],
                [7, 24],
                [8, 66],
                [9, 88], 
                [10, 60]])

    trainSet=[]
    validationSet=[]
    testSet=[]
    for i in range(len(data)):
        if i<7:
            trainSet.append(data[i])
        elif i<9:
            validationSet.append(data[i])
        else:
            testSet.append(data[i])
    trainSet=np.array(trainSet)
    validationSet=np.array(validationSet)
    testSet=np.array(testSet)
    result=""
    result+="Training Set: \n"
    result+=str(trainSet)
    result+="\nValidation Set: "
    result+=str(validationSet)
    result+="\nTest Set: "
    result+=str(testSet)
    result += "\n\n\n----- PURPOSE OF EACH SPLIT -----\n\n"

    result += "Training Set:\n"
    result += "It is used to train the model. The model learns "
    result += "patterns from this data and adjusts its weights.\n\n"

    result += "Validation Set:\n"
    result += "It is used to check and improve model performance "
    result += "during training. We can see how the model performs "
    result += "on unseen data through this set.\n\n"

    result += "Test Set:\n"
    result += "It is used to evaluate the final performance of the "
    result += "model at the end of training. The model does not see "
    result += "test data during training.\n\n"

    result += "----- DATASET SPLIT -----\n"
    result += "Training: 70% -> Model learns\n"
    result += "Validation: 20% -> Model is checked\n"
    result += "Test: 10% -> Final evaluation"
    return result

def task5():
    input=np.array([1, 3, 5, 6, 9], dtype=float)
    target=np.array([2, 4, 6, 8, 10], dtype=float)
    weight=0.0
    bias=0.0
    learnRate=0.01
    epochs=100
    loss=[]
    for e in range(epochs):
        prediction=np.zeros(len(input))
        for i in range(len(input)):
            prediction[i]=input[i]*weight+bias
        total=0
        for i in range(len(target)):
            diff=prediction[i]-target[i]
            total=total+(diff*diff)
        ls=total/len(target)
        loss.append(ls)

        weightGrad=0
        biasGrad=0
        for i in range(len(input)):
            diff=prediction[i]-target[i]
            weightGrad=weightGrad+(diff*input[i])
            biasGrad=biasGrad+diff
        weightGrad=(2/len(input))*weightGrad
        biasGrad=(2/len(input))*biasGrad
        weight=weight-learnRate*weightGrad
        bias=bias-learnRate*biasGrad
    finalPrediction = np.zeros(len(input))
    for i in range(len(input)):
        finalPrediction[i] = input[i] * weight + bias
    result = ""
    result += "----- TRAINING RESULTS -----\n\n"
    result += "Final Weight: " + str(weight) + "\n"
    result += "Final Bias: " + str(bias) + "\n\n"
    result += "----- INPUT / TARGET / PREDICTION -----\n\n"
    for i in range(len(input)):
        result+=("Input: " + str(input[i]) +
            "    Target: " + str(target[i]) +
            "    Prediction: " + str(finalPrediction[i]) + "\n")
    result += "\n----- LOSS DURING TRAINING -----\n\n"
    for e in range(epochs):
        if (e + 1) % 10 == 0:
            result += (
                "Epoch: " + str(e + 1) +
                "    Loss: " + str(loss[e]) + "\n")
    figure=plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs+1), loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss VS Epochs")
    plt.grid(True)
    plt.tight_layout()
    return result, figure

with gr.Blocks() as demo:

    gr.Markdown("# Module 14 — Introduction to Deep Learning")

    with gr.Tab("Task 1 — Basics"):

        gr.Markdown("## AI vs Machine Learning vs Deep Learning")

        output = gr.Textbox(
            label="Task 1 Output",
            lines=30
        )

        button = gr.Button("Show Task 1")

        button.click(
            fn=task1,
            inputs=None,
            outputs=output
        )

    with gr.Tab("Task 2 — Neural Network Basics"):

        gr.Markdown("## Neural Network Components and Data Flow")

        showTask2 = gr.Button("Show Task 2")

        diagramOutput = gr.Image(
            label="Neural Network Diagram"
        )

        flowOutput = gr.Plot(
            label="Flow of Data"
        )

        textOutput = gr.Textbox(
            label="Explanation",
            lines=18
        )

        showTask2.click(
            fn=task2,
            inputs=None,
            outputs=[
                diagramOutput,
                flowOutput,
                textOutput
            ]
        )

    with gr.Tab("Task 3 — Simple Neuron"):

        gr.Markdown("## Simple Neuron using NumPy")

        showTask3 = gr.Button("Run Simple Neuron")

        neuronOutput = gr.Textbox(
            label="Neuron Output",
            lines=10
        )

        showTask3.click(
            fn=task3,
            inputs=None,
            outputs=neuronOutput
        )

    with gr.Tab("Task 4 — Dataset Split"):

        gr.Markdown("## Training, Validation and Test Sets")

        showTask4 = gr.Button("Split Dataset")

        datasetOutput = gr.Textbox(
            label="Dataset Split and Explanation",
            lines=30
        )

        showTask4.click(
            fn=task4,
            inputs=None,
            outputs=datasetOutput
        )

    with gr.Tab("Task 5 — Neural Network Experiment"):

        gr.Markdown("## Basic Neural Network Training")

        runTask5 = gr.Button("Train Neural Network")

        trainingOutput = gr.Textbox(
            label="Training Results",
            lines=25
        )

        lossGraph = gr.Plot(
            label="Loss VS Epochs"
        )

        runTask5.click(
            fn=task5,
            inputs=None,
            outputs=[
                trainingOutput,
                    lossGraph
            ]
        )    
     
demo.launch()