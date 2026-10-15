import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)#固定种子为42,结果可复现
x=np.random.randn(100,1)*5#取100个方差为1的数字*5
true_w,true_b=2.5,1.0
y=true_w*x+true_b+np.random.randn(100,1)*1.5#加入噪声
print(f"x形状:{x.shape},y形状:{y.shape}")
w=np.random.randn(1,1)
b=np.zeros((1,1))
lr=0.01
n_epochs=200
losses=[]
for epoch in range(n_epochs+1):
    y_pred=x@w+b#线性回归
    loss=np.mean((y_pred-y)**2)#求损失 即为方差
    losses.append(loss)
    grad_w=2*np.mean((y_pred-y)*x,axis=0,keepdims=True)#MSE损失函数下对w的求导
    grad_b=2*np.mean(y_pred-y)#对b的求导
    w-=lr*grad_w
    b-=lr*grad_b
    if epoch%40==0:
        print(f"Epoch{epoch:3d}|loss={loss:.4f}|w={w[0,0]:.3f}b={b[0,0]:.3f}")
print(f"\n真实值:w={true_w},b={true_b}")
print(f"预测值:w={w[0,0]:.3f}b={b[0,0]:.3f}")
fig,axes=plt.subplots(1,2,figsize=(12,4))#创建1行，2个子图，返回画布fig和子图数组axes
axes[0].scatter(x,y,s=20,label="数据")#原始数据散点图
axes[0].plot(x,x@w+b,color="red",label="拟合线")
axes[0].set_title("线性回归拟合")
axes[0].legend()#显示图例
axes[1].plot(losses)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("loss")
axes[1].set_title("损失下降曲线")
plt.tight_layout()#自动调整间距
plt.show()
#MSE均方误差 loss=1/n(y预测-y真实)**2 求导为2*loss