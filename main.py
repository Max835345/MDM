import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Множество точек H
H = np.random.rand(10, 2)
m, n = H.shape

# Целевая функция: ||H.T @ p||^2 = (H.T @ p)^T (H.T @ p)
def objective(p):
    v = H.T @ p
    return np.dot(v, v)

# Ограничения:
constraints = [
    {'type': 'eq', 'fun': lambda p: np.sum(p) - 1}  # сумма весов = 1
]
bounds = [(0, 1) for _ in range(m)]  # веса >= 0

# Начальное приближение — равномерное
p0 = np.ones(m) / m

# Решение задачи
result = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=constraints)

# Итоговая точка
p_opt = result.x
v_opt = H.T @ p_opt

# Визуализация
plt.scatter(H[:, 0], H[:, 1], color='blue', label='H')
plt.scatter(0, 0, color='black', label='Начало координат')
plt.scatter(v_opt[0], v_opt[1], color='green', label='Решение')

# Выпуклая оболочка
hull = ConvexHull(H)
for simplex in hull.simplices:
    plt.plot(H[simplex, 0], H[simplex, 1], 'k-')

# Вектор к решению
plt.arrow(0, 0, v_opt[0], v_opt[1], head_width=0.02, color='gray', length_includes_head=True)

plt.title(f"Ближайшая точка к началу: ||v|| = {np.linalg.norm(v_opt):.4f}")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
