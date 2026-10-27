import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
np.random.seed(42)
def make_moon(n,center,radius=3.0,noise=0.3):
    theta=np.random.rand(n)*2*np.pi#固定角度
    r=radius+np.random.randn(n)*noise
    return np.c_[center[0]+r*np.cos(theta),center[1]+r*np.sin(theta)]
inner =make_moon(150,[0,0],radius=2.0)#内圈
outer=make_moon(150,[0,0],radius=4.0)#外圈
x=np.vstack([inner,outer]).astype(np.float32)
y=np.array([0]*150+[1]*150,dtype=np.int64)
x_t=torch.from_numpy(x)#共享内存嵌套torch外壳用于张量计算
y_t=torch.from_numpy(y)
print("x_t shape:",x_t.shape,"y_t shape:",y_t.shape)
class TwoLayerNet(nn.Module):
    def __init__(self,n_features=2,n_hidden=4,n_classes=2):
        super().__init__()
        self.fc1=nn.Linear(n_features,n_hidden)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(n_hidden,n_classes)
    def forward(self,x):#xw+b
        x=self.fc1(x)#300,4输入x为300,2->z1
        x=self.relu(x)#300,4->a
        x=self.fc2(x)#300,2->z2
        return x
net=TwoLayerNet()
print(net)
criterion=nn.CrossEntropyLoss()#多分类交叉熵CE
optimizer=torch.optim.SGD(net.parameters(),lr=0.1) #parameters拿到全部w,b SGD随机梯度下降
n_epochs=3000
losses=[]
for epoch in range(n_epochs+1):
    scores=net(x_t)#正向传播
    loss=criterion(scores,y_t)#计算代价
    optimizer.zero_grad()#清空梯度
    loss.backward()#自动求导
    optimizer.step()#梯度下降
    losses.append(loss.item())
    if epoch%500==0:
        with torch.no_grad():
            probs=torch.softmax(scores,dim=1)
            preds=probs.argmax(dim=1)
            acc=(preds==y_t).float().mean()
            print("Epoch %4d | loss=%.4f|acc=%.3f"%(epoch,loss.item(),acc))
    #画图
def plot_boundary(net, X):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_t = torch.from_numpy(grid)

    with torch.no_grad():
        probs = torch.softmax(net(grid_t), dim=1)
        labels = (probs[:, 1] > 0.5).numpy().reshape(xx.shape).astype(int)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, labels, levels=2, cmap="Pastel1", alpha=0.7)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", s=12, label="inner")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="orange", s=12, label="outer")
    plt.title("PyTorch 2-layer net decision boundary")
    plt.legend()
    plt.savefig("day5_boundary.png")
    plt.close()

plot_boundary(net, x)

plt.figure(figsize=(6, 3))
plt.plot(losses)
plt.title("training loss")
plt.xlabel("epoch")
plt.savefig("day5_loss.png")
plt.close()

print("done. figures saved.")


