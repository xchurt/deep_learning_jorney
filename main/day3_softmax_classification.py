import numpy as np
import matplotlib.pyplot as plt
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
