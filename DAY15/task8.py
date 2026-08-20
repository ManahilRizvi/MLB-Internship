import numpy as np
import matplotlib.pyplot as plt
x=np.array([[1, 2],
            [2, 3],
            [3, 4],
            [4, 5]], dtype=float)
y=np.array([[3], [5], [7], [9]], dtype=float)
def relu(x):
    return np.maximum(0, x)

def reluDeriv(x):
    derivative=np.zeros_like(x)
    derivative[x>0]=1
    return derivative

def mseLoss(y, pred):
    diff=y-pred
    loss=np.mean(diff*diff)
    return loss

np.random.seed(42)
w1Ini=np.random.randn(2, 4)*0.1
b1Ini=np.zeros((1, 4))
w2Ini=np.random.randn(4, 1)*0.1
b2Ini=np.zeros((1, 1))

def trainModel(optimName):
    w1=w1Ini.copy()
    b1=b1Ini.copy()
    w2=w2Ini.copy()
    b2=b2Ini.copy()

    vw1=np.zeros_like(w1)
    vb1=np.zeros_like(b1)
    vw2=np.zeros_like(w2)
    vb2=np.zeros_like(b2)

    mw1=np.zeros_like(w1)
    mb1=np.zeros_like(b1)
    mw2=np.zeros_like(w2)
    mb2=np.zeros_like(b2)

    sw1=np.zeros_like(w1)
    sb1=np.zeros_like(b1)
    sw2=np.zeros_like(w2)
    sb2=np.zeros_like(b2)

    learnRate=0.01
    momentum=0.9
    beta1=0.9
    beta2=0.999
    epsilon=1e-8
    loss=[]

    for epoch in range(1000):
        z1=np.dot(x, w1)+b1
        a1=relu(z1)
        z2=np.dot(a1, w2)+b2
        pred=z2

        ls=mseLoss(y, pred)
        loss.append(ls)

        dz2=(2/len(y))*(pred-y)
        dw2=np.dot(a1.T, dz2)
        db2=np.sum(dz2, axis=0, keepdims=True)
        da1=np.dot(dz2, w2.T)
        dz1=da1*reluDeriv(z1)
        dw1=np.dot(x.T, dz1)
        db1=np.sum(dz1, axis=0, keepdims=True)

        if optimName=="SGD":
            w1=w1-learnRate*dw1
            b1=b1-learnRate*db1
            w2=w2-learnRate*dw2
            b2=b2-learnRate*db2

        elif optimName=="Momentum":
            vw1=momentum*vw1+dw1
            vb1=momentum*vb1+db1
            vw2=momentum*vw2+dw2
            vb2=momentum*vb2+db2

            w1=w1-learnRate*vw1
            b1=b1-learnRate*vb1
            w2=w2-learnRate*vw2
            b2=b2-learnRate*vb2

        elif optimName=="Adam":
            mw1=beta1*mw1+(1-beta1)*dw1
            mb1=beta1*mb1+(1-beta1)*db1
            mw2=beta1*mw2+(1-beta1)*dw2
            mb2=beta1*mb2+(1-beta1)*db2

            sw1=beta2*sw1+(1-beta2)*(dw1*dw1)
            sb1=beta2*sb1+(1-beta2)*(db1*db1)
            sw2=beta2*sw2+(1-beta2)*(dw2*dw2)
            sb2=beta2*sb2+(1-beta2)*(db2*db2)

            mw1Corr=mw1/(1-beta1**(epoch+1))
            mb1Corr=mb1/(1-beta1**(epoch+1))
            mw2Corr=mw2/(1-beta1**(epoch+1))
            mb2Corr=mb2/(1-beta1**(epoch+1))

            sw1Corr=sw1/(1-beta2**(epoch+1))
            sb1Corr=sb1/(1-beta2**(epoch+1))
            sw2Corr=sw2/(1-beta2**(epoch+1))
            sb2Corr=sb2/(1-beta2**(epoch+1))

            w1=w1-learnRate*(mw1Corr/(np.sqrt(sw1Corr)+epsilon))
            b1=b1-learnRate*(mb1Corr/(np.sqrt(sb1Corr)+epsilon))
            w2=w2-learnRate*(mw2Corr/(np.sqrt(sw2Corr)+epsilon))
            b2=b2-learnRate*(mb2Corr/(np.sqrt(sb2Corr)+epsilon))
    return loss, w1, b1, w2, b2

sgdLoss, w1Sgd, b1Sgd, w2Sgd, b2Sgd=trainModel("SGD")
momLoss, w1Mome, b1Mome, w2Mome, b2Mome=trainModel("Momentum")
adamLoss, w1Adam, b1Adam, w2Adam, b2Adam=trainModel("Adam")
print("SGD Loss: ", sgdLoss[-1])
print("Momentum Loss: ", momLoss[-1])
print("Adam Loss: ", adamLoss[-1])

plt.plot(sgdLoss, label="SGD")
plt.plot(momLoss, label="SGD with Momentum")
plt.plot(adamLoss, label="Adam")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Optimizer Comparison")
plt.legend()
plt.show()