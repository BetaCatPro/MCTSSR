import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(1, 256, num=256)
y = np.log(x).reshape(-1,1) + np.random.uniform(-0.3,0.5,[256,1])

lr = LinearRegression()
lr.fit(x.reshape(-1,1),y.reshape(-1,1))
pre_y1 = lr.predict(x.reshape(-1,1))

svr = SVR()
svr.fit(x.reshape(-1,1),y.reshape(-1,1))
pre_y2 = svr.predict(x.reshape(-1,1))

fig, ax = plt.subplots()
ax.plot(x, pre_y1, linewidth=3, linestyle='-', label='线性回归')
ax.plot(x, pre_y2, linewidth=3, linestyle='-', label='非线性回归')

# ax.fill_between(x, pre_y2 - 0.1, pre_y2 + 0.1,alpha=0.4, facecolor='r')

ax.plot(x, y, '.')

ax.grid(linestyle='-.')

plt.legend()

plt.show()

# plt.savefig('test.pdf')