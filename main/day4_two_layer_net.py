import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def make_moon(n,center,radius=3.0,noise=0.3):#构造数据
    theta=np.random.rand(n)*2*np.pi#固定角度在0,2pi
    r=radius+np.random.randn(n)*noise#基础半径+高斯噪声
    return np.c_[center[0]+r*np.cos(theta),center[1]+r*np.sin(theta)]
inner=make_moon(150,[0,0],radius=2.0)#半径2 150样本
outer=make_moon(150,[0,0],radius=4.0)#半径4 150个样本
x=np.vstack([inner,outer])#拼接样本坐标 竖直拼接
y=np.array([0]*150+[1]*150)#标签
y_onehot=np.eye(2)[y]#onehot编码
print(f"x形状:{x.shape},y_onehot形状:{y_onehot.shape}")
def relu(z):
    return np.maximum(0,z)
def relu_deriv(z):#relu激活函数的导数
    return (z>0).astype(float)
def sigmoid(z):
    return 1/(1+np.exp(-z))
n_features=2#特征数
n_hidden=4#神经元数
n_classes=2#0 1分类
w1=np.random.randn(n_features,n_hidden)*0.5#输入>隐藏的权重矩阵 *0.5防止饱和
b1=np.zeros((1,n_hidden))#偏置
w2=np.random.randn(n_hidden,n_classes)*0.5#隐藏层>输出层
b2=np.zeros((1,n_classes))#偏置
lr=0.1
n_epocs=3000
losses=[]
for epoch in range(n_epocs+1):
    #前向传播
    z1=x@w1+b1#隐藏层预激活
    a1=relu(z1)#过ReLu激活
    z2=a1@w2+b2#输出层预激活
    probs=sigmoid(z2)#输出通道概率
    loss=-np.mean(np.sum(y_onehot * np.log(probs+1e-8),axis=1))
    losses.append(loss)
    #反向传播
    d_z2=probs-y_onehot#预测概率-onehot标签,合并导数
    d_w2=(a1.T@d_z2)/len(x)
    d_b2=np.mean(d_z2,axis=0)#偏置梯度，对样本维度求平均
    #回传隐藏层
    d_a1=d_z2@w2.T
    d_z1=d_a1*relu_deriv(z1)
    d_w1=(x.T@d_z1)/len(x)
    d_b1=np.mean(d_z1,axis=0)
    #梯度下降
    w2-=lr*d_w2;b2-=lr*d_b2
    w1-=lr*d_w1;b1-=lr*d_b1
    if epoch%500==0:
        acc = np.mean((probs[:,1]>0.5)==y)
        print(f"Epoch{epoch:4d}|loss={loss:.4f}|acc={acc:.3f}")
#画图观察
def plot_boundary(X,y,W1,b1,W2,b2):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    a1 = relu(grid @ W1 + b1)
    probs = sigmoid(a1 @ W2 + b2)
    labels = (probs[:, 1] > 0.5).reshape(xx.shape).astype(int)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, labels, levels=2, cmap="Pastel1", alpha=0.7)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=12, label="内圈")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='orange', s=12, label="外圈")
    plt.title("神经网络决策边界（非线性）")
    plt.legend()
    plt.show()
plot_boundary(x, y, w1, b1, w2, b2)
plt.figure(figsize=(6, 3))
plt.plot(losses)
plt.title("训练损失")
plt.xlabel("Epoch")
plt.show()
#二层全连接神经网络 输入>隐藏relU>输出sigmoid
