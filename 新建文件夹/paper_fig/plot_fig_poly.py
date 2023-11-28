import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(10, 256, num=128)
X = x.reshape(-1, 1)
y = np.log(x).reshape(-1,1) + np.random.uniform(-0.3,0.2,[128,1])

poly = PolynomialFeatures(degree=7)
poly.fit(X)
X2 = poly.transform(X)

reg = LinearRegression()
reg.fit(X2, y)
y_predict = reg.predict(X2)

tmp_random = np.random.randint(0,127, 50)


plt.scatter(x[tmp_random], y[tmp_random], marker = 'o', c='g', label='有标记数据')
plt.scatter(np.delete(x, tmp_random), np.delete(y, tmp_random), marker = '*', c='m', label='无标记数据')


plt.plot(np.sort(x), y_predict[np.argsort(x)], color='orange', linewidth=3, linestyle='-', label='回归模型')

plt.grid(linestyle='-.')
plt.legend()
plt.show()