import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Генерация случайных точек
H = np.random.rand(5, 2)
m, n = H.shape

# Инициализация
v = np.zeros(n)
p = np.ones(m) / m
path = [v.copy()]

# МДМ-алгоритм
for _ in range(1000):
    scalars = H @ v
    delta = np.max(scalars) - np.min(scalars)
    if delta < 1e-6:
        break
    i_max = np.argmax(scalars)
    i_min = np.argmin(scalars)
    v = v - p[i_max] * (H[i_max] - H[i_min])
    p[i_min] += p[i_max]
    p[i_max] = 0
    path.append(v.copy())

# Визуализация
path = np.array(path)
plt.plot(path[:,0], path[:,1], 'ro--')
plt.scatter(H[:,0], H[:,1], c='blue', label='H')
plt.scatter(0, 0, c='black', label='Начало координат')
plt.scatter(path[-1,0], path[-1,1], c='green', label='Решение')
hull = ConvexHull(H)
for simplex in hull.simplices:
    plt.plot(H[simplex, 0], H[simplex, 1], 'k-')
plt.legend()
plt.title("МДМ-метод")
plt.axis('equal')
plt.grid(True)
plt.show()