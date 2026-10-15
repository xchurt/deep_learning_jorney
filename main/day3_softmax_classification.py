import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
def make_blob(n,center,noise=0.6):#生成n个中心点center,噪声noise
    return np.random.randn(n,2)*noise+np.array(center)
c0=make_blob(80,[-3,-3])
c1=make_blob(80,[3,-3])
c2=make_blob(80,[0,3])
x=np.vstack([c0,c1,c2])
y=np.array([0]*80+[1]*80+[2]*80)
y_onehot=np.eye(3)[y]
print(f"形状{x.shape}y_onehot形状:{y_onehot.shape}")
def softmax(z):
    z_shift=z-np.max(z,axis=1,keepdims=True)
    exp_z=np.exp(z_shift)#取e的z_shift次方
    return exp_z/np.sum(exp_z,axis=1,keepdims=True)
n_features=x.shape[1]
n_classes=3
w=np.random.randn(n_features,n_classes)*0.01
b=np.zeros((1,n_classes))
lr=0.1
n_epochs=1000
losses=[]
for epoch in range(n_epochs+1):
    z=x@w+b
    probs=softmax(z)
    loss=-np.mean(np.sum(y_onehot*np.log(probs+1e-8),axis=1))
    losses.append(loss)
    grad_w=(x.T@(probs-y_onehot))/len(x)
    grad_b=np.mean(probs-y_onehot,axis=0)
    w-=lr*grad_w
    b-=lr*grad_b
    if epoch%200==0:
        preds=np.argmax(probs,axis=1)
        acc=np.mean(preds==y)
        print(f"Epoch{epoch:4d}|loss={loss:.4f}|acc={acc:.3f}")
print(f"\n最终w:\n{w}")
print(f"最终b:{b}")
def plot_boundary(x,y,w,b):
    x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
    y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
    xx,yy=np.meshgrid(np.linspace(x_min,x_max,200),np.linspace(y_min,y_max,200))
    grid=np.c_[xx.ravel(),yy.ravel()]
    probs=softmax(grid@w+b)
    labels=np.argmax(probs,axis=1).reshape(xx.shape)
    plt.figure(figsize=(6,5))
    plt.contourf(xx,yy,labels,levels=3,cmap="Pastel1",alpha=0.7)
    colors=['blue','orange','green']
    for i in range(3):
        mask=y==i
        plt.scatter(x[mask,0],x[mask,1],c=colors[i],s=15,label=f"类别{i}")
    plt.title("Softmax 三分类决策边界")
    plt.legend()
    plt.show()
plot_boundary(x,y,w,b)
plt.figure(figsize=(6,3))
plt.plot(losses)
plt.title("多分类交叉熵损失")
plt.xlabel("Epoch")
plt.show()