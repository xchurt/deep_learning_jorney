import torch
import torch.nn as nn
import torch.optim as optim
from networkx.algorithms.coloring.greedy_coloring import strategy_independent_set
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
transform=transforms.Compose([#数据预处理 多个图像变换顺序拼接
    transforms.ToTensor(),#0-255的PIL图片转为pytorch张量缩放到0,1 维度从(H,W）到(C,H,W)
    transforms.Normalize((0.1307,), (0.3081,))#(均值，标准差)
])#数据归一化+数据分布中心化
#数据加载
train_dataset=datasets.MNIST(root="./data",#数据存储地址
                             train=True,#1训练集 0测试集
                             download=True,#自动下载开关
                             transform=transform#使用前面定义的数据预处理
)
test_dataset=datasets.MNIST(root="./data",
                            train=False,
                            download=True,
                            transform=transform
)

print("训练集数量:",len(train_dataset))
print("测试集数量:",len(test_dataset))
print("每张图片的形状",train_dataset[0][0].shape)#
batch_size=64#一次丢入网络的图个数
train_loader=DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True#数据打乱开关
                        )
test_loader=DataLoader(test_dataset,
                       batch_size=batch_size,
                       shuffle=False
)
images,labels=next(iter(train_loader))#iter将train_loader变为迭代器给next拿出
print("一个批次的图片形状",images.shape)
print("一个批次的标签形状",labels.shape)
class MnistNet(nn.Module):#神经网络
    def __init__(self):
        super().__init__()#调用父类nn.Module初始化
        self.fc1=nn.Linear(28*28,128)#输入28*28的像素 映射到128个神经元(隐藏层)
        self.relu=nn.ReLU()#激活函数 加入非线性
        self.fc2=nn.Linear(128,10)#隐藏128的神经元输出10个值0~9
    def forward(self,x):#前向传播
        x=x.view(x.size(0),-1)#reshape变形铺成一维向量
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x
net=MnistNet()#创建模型
print(net)#输出网络结构
criterion=nn.CrossEntropyLoss()#CE多分类损失函数
optimizer=optim.Adam(net.parameters(),lr=0.001)#Adam自适应学习率优化算法
n_epochs=5
train_losses=[]
for epoch in range(n_epochs):
    running_loss=0.0
    net.train()#训练模式
    for batch_idx, (images,labels) in enumerate(train_loader):
        scores=net(images)#前向传播
        loss=criterion(scores,labels)#计算损失
        optimizer.zero_grad()#清空梯度
        loss.backward()#反向传播
        optimizer.step()#优化器获取梯度 更新权重
        running_loss+=loss.item()
        if batch_idx % 200 == 0:
            print("Epoch %d|batch %4d|loss=%4f"%(epoch,batch_idx,loss.item()))
    avg_loss=running_loss/len(train_loader)
    train_losses.append(avg_loss)
    print(">>>Epoch %d 平均损失 %.4f"%(epoch,avg_loss))
#可视化
net.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        outputs=net(images)
        preds=outputs.argmax(dim=1)
        total+=labels.size(0)
        correct+=(preds==labels).sum().item()
print("\n测试集准确率:%.2f%%"%(100.0*correct/total))
net.eval()
images, labels = next(iter(test_loader))
with torch.no_grad():
    preds = net(images).argmax(dim=1)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    ax = axes[i // 5][i % 5]
    ax.imshow(images[i].squeeze(), cmap="gray")
    ax.set_title("pred:%d true:%d" % (preds[i], labels[i]),
                 color="green" if preds[i] == labels[i] else "red")
    ax.axis("off")
plt.savefig("day6_predictions.png")
plt.close()

plt.figure(figsize=(6, 3))
plt.plot(train_losses, marker="o")
plt.title("MNIST training loss")
plt.xlabel("epoch")
plt.savefig("day6_loss.png")
plt.close()

print("done. figures saved.")