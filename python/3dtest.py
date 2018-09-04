from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(10)
y = np.arange(10)
z = np.random.standard_normal(10)
c = np.random.standard_normal(10)

ax.scatter(x, y, z, c=c, cmap=plt.hot())
plt.show()