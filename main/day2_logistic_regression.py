import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)#固定随机种子，可以复现
c0=np.random.randn(100,2)+np.array([-2,-2])
c1=np.random.randn(100,2)+np.array([2,2])
x=np.vstack([c0,c1])
y=np.vstack([np.zeros((100,1)),np.ones((100,1))])
print(f"x形状:{x.shape},y形状:{y.shape}")
def sigmoid(z):
    return 1/(1+np.exp(-z))
n_features=x.shape[1]
w=np.random.randn(n_features,1)*0.01
b=np.zeros((1,1))
lr=0.1
n_epochs=1000#训练次数
losses=[]
for epoch in range(n_epochs+1):
    z=x@w+b
    y_prob=sigmoid(z)
    loss=-np.mean(y*np.log(y_prob+1e-8)+(1-y)*np.log(1-y_prob+1e-8))#加1e-8防止log0出现NaN
    losses.append(loss)
    grad_w=(x.T@(y_prob-y))/len(y)
    grad_b=np.mean(y_prob-y)
    w-=lr*grad_w
    b-=lr*grad_b
    if epoch%200==0:
        acc=np.mean((y_prob>0.5)==y)
        print(f"Epoch{epoch:4d}|loss={loss:.4f}|acc={acc:.3f}")
print(f"\n最终权重w:\n{w}")
print(f"最终偏置b:{b[0,0]:.4f}")
def plot_decision_boundary(x,y,w,b):
    x_min,x_max=x[:,0].min()-1,x[:,0].max()+1#向外扩大1单位，绘图范围
    y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
    xx,yy=np.meshgrid(np.linspace(x_min,x_max,100),np.linspace(y_min,y_max,100))
    grid=np.c_[xx.ravel(),yy.ravel()]
    probs=sigmoid(grid@w+b).reshape(xx.shape)
    plt.figure(figsize=(6,5))
    plt.contourf(xx,yy,probs,levels=50,cmap="RdBu",alpha=0.6)
    plt.colorbar(label="P(类别1)")
    plt.scatter(c0[:,0],c0[:,1],c="blue",s=20,label="类别0")
    plt.scatter(c1[:,0],c1[:,1],c="red",s=20,label="类别1")
    plt.title("逻辑回归决策的边界")
    plt.legend()
    plt.show()
plot_decision_boundary(x,y,w,b)
plt.figure(figsize=(6,3))
plt.plot(losses)
plt.title("交叉熵损失下降")
plt.xlabel("Epoch")
plt.show()
#BCE二元交叉熵 loss=-ylog(y预测)-(1-y)log(1-y预测)
#求导为(y预-y)