import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fn(xy):
    x, y = xy
    return np.exp(-(5**2-(x**2+y**2))**2/250 + xy[1]/10) * (7./4-np.sin(7*np.arctan2(x, y)))

plt.figure(figsize=[6,6])
mx, my = np.meshgrid(np.linspace(-10, 10, 101), np.linspace(-10, 10, 101))
mz = fn([mx,my])
ax = plt.axes([0,0,1,1], projection='3d')
ax.plot_surface(mx,my,mz,rstride=2, alpha=0.2, edgecolor='k', cmap='rainbow')
plt.show()

xy0 = np.array([3,-3])
bound = np.array([[-6,6], [-6,6]])
s = (bound[:,1]-bound[:,0])/10.
n = 16000
xy = []
p = []
p0 = fn(xy0)

for i in range(n):
    idou = np.random.normal(0,s,2)
    hazure = (xy0+idou<bound[:,0])|(xy0+idou>bound[:,1])
    while(np.any(hazure)):
        idou[hazure] = np.random.normal(0,s,2)[hazure] # 外れたものだけもう一度ランダムする
        hazure = (xy0+idou<bound[:,0])|(xy0+idou>bound[:,1])
    xy1 = xy0 + idou # 新しい位置の候補
    p1 = fn(xy1) # 新しい位置の確率
    r = p1/p0 # 新しい位置と現在の位置の確率の比率
    # 比率は1より高い場合は常に移動するが、低い場合は確率で移動する
    if(r>1 or r>np.random.random()):
        xy0 = xy1 # 現在の位置を新しい位置に移動する
        p0 = p1
        xy.append(xy0) # 新しい位置を格納
        p.append(p0) # 新しい確率を格納

xy = np.stack(xy)
x,y = xy[:,0],xy[:,1]
plt.figure(figsize=[7,6])
plt.gca(aspect=1)
plt.scatter(x,y,c=p,alpha=0.1,edgecolor='k')
plt.colorbar(pad=0.01)
plt.scatter(*xy[np.argmax(p)],s=150,c='r',marker='*',edgecolor='k') # 最大値を星で示す
plt.tight_layout()
plt.show()